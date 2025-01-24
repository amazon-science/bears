"""Jupyter-friendly asyncio usage:"""

import asyncio
import atexit
import inspect
import threading
from functools import partial
from typing import Optional


def _asyncio_start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


## Async wrapper to run a synchronous or asynchronous function in the event loop
async def __run_fn_async(fn, *args, run_sync_in_executor: bool = True, **kwargs):
    if inspect.iscoroutinefunction(fn):
        ## If fn is defined with `def async`, run this using asyncio mechanism,
        ## meaning code inside fn is run in an sync way, except for the "await"-marked lines, which will
        ## be run asynchronously. Note that "await"-marked lines must call other functions defined using "def async".
        result = await fn(*args, **kwargs)
    else:
        ## The function is a regular synchronous function.
        if run_sync_in_executor:
            ## Run in the default executor (thread pool) for the event loop, otherwise it blocks the event loop
            ## until the function execution completes.
            ## The executor lives for the lifetime of the event loop. Ref: https://stackoverflow.com/a/33399896/4900327
            ## This basically is the same as run_concurrent, but with no control on the number of threads.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, partial(fn, *args, **kwargs))
        else:
            ## Run the synchronous function directly in the event loop.
            ## This will block the event loop until the function execution is complete,
            ## preventing other tasks from running during this time.
            result = fn(*args, **kwargs)
    return result


_ASYNCIO_EVENT_LOOP = None
_ASYNCIO_EVENT_LOOP_THREAD = None


## Function to submit the coroutine to the asyncio event loop
def run_asyncio(
    fn,
    *args,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    **kwargs,
):
    global _ASYNCIO_EVENT_LOOP
    global _ASYNCIO_EVENT_LOOP_THREAD
    if event_loop is None:
        ## Create a new loop and a thread running this loop
        if _ASYNCIO_EVENT_LOOP is None:
            _ASYNCIO_EVENT_LOOP = asyncio.new_event_loop()
        if _ASYNCIO_EVENT_LOOP_THREAD is None:
            _ASYNCIO_EVENT_LOOP_THREAD = threading.Thread(
                target=_asyncio_start_event_loop,
                args=(_ASYNCIO_EVENT_LOOP,),
            )
            _ASYNCIO_EVENT_LOOP_THREAD.start()

        event_loop: asyncio.AbstractEventLoop = _ASYNCIO_EVENT_LOOP
    ## Create a coroutine (i.e. Future), but do not actually start executing it.
    coroutine = __run_fn_async(fn, *args, **kwargs)
    ## Schedule the coroutine to execute on the event loop
    ## (which is running on thread _ASYNCIO_EVENT_LOOP_THREAD).
    return asyncio.run_coroutine_threadsafe(coroutine, event_loop)


async def async_http_get(url):
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def _cleanup_event_loop():
    global _ASYNCIO_EVENT_LOOP
    global _ASYNCIO_EVENT_LOOP_THREAD

    if _ASYNCIO_EVENT_LOOP is not None and _ASYNCIO_EVENT_LOOP_THREAD is not None:
        if _ASYNCIO_EVENT_LOOP.is_running():
            _ASYNCIO_EVENT_LOOP.call_soon_threadsafe(_ASYNCIO_EVENT_LOOP.stop)
        _ASYNCIO_EVENT_LOOP_THREAD.join()
        _ASYNCIO_EVENT_LOOP.close()
        del _ASYNCIO_EVENT_LOOP
        del _ASYNCIO_EVENT_LOOP_THREAD


## Register the cleanup function to be called upon Python program exit
atexit.register(_cleanup_event_loop)
