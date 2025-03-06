import builtins
import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures._base import Executor
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import ConfigDict, model_validator

from bears.constants import Parallelize
from bears.util.language import (
    Alias,
    Parameters,
    ProgressBar,
    as_tuple,
    filter_kwargs,
    get_default,
    is_dict_like,
    is_list_or_set_like,
    type_str,
)

from ._asyncio import run_asyncio
from ._processes import ActorPoolExecutor, ActorProxy, run_parallel
from ._ray import RayPoolExecutor, run_parallel_ray
from ._threads import (
    RestrictedConcurrencyThreadPoolExecutor,
    kill_thread,
    run_concurrent,
    suppress_ThreadKilledSystemException,
)
from ._utils import (
    _LOCAL_ACCUMULATE_ITEM_WAIT,
    _LOCAL_ACCUMULATE_ITER_WAIT,
    _RAY_ACCUMULATE_ITEM_WAIT,
    _RAY_ACCUMULATE_ITER_WAIT,
    accumulate_iter,
)


def worker_ids(
    executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, ActorPoolExecutor]],
) -> Set[int]:
    ## Returns a set of unique identifiers for all workers in the given executor
    ## Input: executor - any supported pool executor (Thread, Process, or Actor)
    ## Output: Set of thread IDs or process IDs depending on executor type

    if isinstance(executor, ThreadPoolExecutor):
        ## For thread pools, return set of thread identifiers
        return {th.ident for th in executor._threads}
    elif isinstance(executor, ProcessPoolExecutor):
        ## For process pools, return set of process IDs
        return {p.pid for p in executor._processes.values()}
    elif isinstance(executor, ActorPoolExecutor):
        ## For actor pools, return set of actor process IDs
        return {_actor._process.pid for _actor in executor._actors}

    ## Raise error if executor type is not supported
    raise NotImplementedError(f"Cannot get worker ids for executor of type: {executor}")


class ExecutorConfig(Parameters):
    """
    Configuration class for parallel execution settings used by dispatch functions.
    Provides a structured way to define parallelization strategy and execution constraints.

    Attributes:
        parallelize: Type of parallelization to use (sync, threads, processes, ray)
        max_workers: Maximum number of parallel workers (None uses system defaults)
        max_calls_per_second: Rate limiting for execution calls (infinity means no limit)

    Example usage:
        >>> config = ExecutorConfig(
                parallelize='threads',
                max_workers=4,
                max_calls_per_second=100.0
            )
        >>> executor = dispatch_executor(config=config)

        # Using with num_workers alias
        >>> config = ExecutorConfig(
                parallelize='processes',
                num_workers=8  # alias for max_workers
            )
    """

    model_config = ConfigDict(extra="ignore")  ## Silently ignore any extra parameters for flexibility

    parallelize: Parallelize
    max_workers: Optional[int] = None  ## None lets the executor use system-appropriate defaults
    max_calls_per_second: float = float("inf")  ## No rate limiting by default

    @model_validator(mode="before")
    @classmethod
    def _set_params(cls, params: Dict) -> Dict:
        """
        Pre-processes configuration parameters to support alternate parameter names.
        Set various aliases of 'max_workers' for compatibility.
        """
        Alias.set_num_workers(params, param="max_workers")
        return params


def dispatch(
    fn: Callable,
    *args,
    parallelize: Parallelize,
    forward_parallelize: bool = False,
    delay: float = 0.0,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Any:
    parallelize: Parallelize = Parallelize.from_str(parallelize)
    if forward_parallelize:
        kwargs["parallelize"] = parallelize
    time.sleep(delay)
    if parallelize is Parallelize.sync:
        return fn(*args, **kwargs)
    elif parallelize is Parallelize.asyncio:
        return run_asyncio(fn, *args, **kwargs)
    elif parallelize is Parallelize.threads:
        return run_concurrent(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.processes:
        return run_parallel(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.ray:
        return run_parallel_ray(fn, *args, executor=executor, **kwargs)
    raise NotImplementedError(f"Unsupported parallelization: {parallelize}")


def dispatch_executor(
    *, config: Optional[Union[ExecutorConfig, Dict]] = None, **kwargs
) -> Optional[Executor]:
    """
    Creates and configures an executor based on the provided configuration settings.
    Returns None for synchronous execution or when using default system executors.

    The executor handles parallel task execution with configurable constraints like
    maximum workers and rate limiting for thread-based execution.

    Args:
        config: ExecutorConfig instance or dict containing parallelization settings
        **kwargs: Additional configuration parameters that override config values

    Returns:
        Configured executor instance or None if using defaults/sync execution

    Example usage:
        >>> config = ExecutorConfig(
                parallelize='threads',
                max_workers=4,
                max_calls_per_second=100.0
            )
        >>> executor = dispatch_executor(config=config)

        >>> executor = dispatch_executor(
                config=dict(parallelize='processes', max_workers=8)
            )
    """
    if config is None:
        config: Dict = dict()
    else:
        assert isinstance(config, ExecutorConfig)
        config: Dict = config.model_dump(exclude=True)

    ## Merge passed kwargs with config dict to allow parameter overrides
    config: ExecutorConfig = ExecutorConfig(**{**config, **kwargs})

    if config.max_workers is None:
        ## Return None to use system defaults - this is more efficient for simple cases
        return None

    if config.parallelize is Parallelize.sync:
        return None
    elif config.parallelize is Parallelize.threads:
        ## Use restricted concurrency for threads to enable rate limiting
        return RestrictedConcurrencyThreadPoolExecutor(
            max_workers=config.max_workers,
            max_calls_per_second=config.max_calls_per_second,
        )
    elif config.parallelize is Parallelize.processes:
        ## Actor-based pool enables better control over process lifecycle
        return ActorPoolExecutor(
            max_workers=config.max_workers,
        )
    elif config.parallelize is Parallelize.ray:
        ## Ray executor for distributed execution across multiple machines
        return RayPoolExecutor(
            max_workers=config.max_workers,
        )
    else:
        raise NotImplementedError(
            f"Unsupported: you cannot create an executor with {config.parallelize} parallelization."
        )


def stop_executor(
    executor: Optional[Executor],
    force: bool = True,  ## Forcefully terminate, might lead to work being lost.
):
    if executor is not None:
        if isinstance(executor, ThreadPoolExecutor):
            suppress_ThreadKilledSystemException()
            if force:
                executor.shutdown(wait=False)  ## Cancels pending items
                for tid in worker_ids(executor):
                    kill_thread(tid)  ## Note; after calling this, you can still submit
                executor.shutdown(wait=False)  ## Note; after calling this, you cannot submit
            else:
                executor.shutdown(wait=True)
            del executor
        elif isinstance(executor, ProcessPoolExecutor):
            if force:
                for process in executor._processes.values():  # Internal Process objects
                    process.terminate()  # Forcefully terminate the process

                # Wait for the processes to clean up
                for process in executor._processes.values():
                    process.join()
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            del executor
        elif isinstance(executor, ActorPoolExecutor):
            for actor in executor._actors:
                assert isinstance(actor, ActorProxy)
                actor.stop(cancel_futures=force)
                del actor
            del executor


def map_reduce(
    struct: Union[List, Tuple, np.ndarray, pd.Series, Set, frozenset, Dict, Iterator],
    *args,
    fn: Callable,
    parallelize: Parallelize,
    forward_parallelize: bool = False,
    item_wait: Optional[float] = None,
    iter_wait: Optional[float] = None,
    iter: bool = False,
    reduce_fn: Optional[Callable] = None,
    worker_queue_len: Optional[int] = 5,
    **kwargs,
) -> Optional[Union[Any, Generator]]:
    """
    Applies a function to batches of elements in a data structure in parallel.
    Processes data in batches to manage memory usage.

    This is particularly useful for processing large datasets where you want to:
    1. Control memory usage by only keeping a number of batches of data in memory at once
    2. Take advantage of parallel execution for efficient processing
    3. Track progress of both batch submission and results collection

    Args:
        struct: Input data structure to iterate over (list-like, dict-like, or iterator)
        *args: Additional positional args passed to each fn call
        fn: Function to apply to each element in the batch
        parallelize: Execution strategy (sync, threads, processes, ray, asyncio)
        batch_size: Number of items to process in each batch (memory management)
        forward_parallelize: If True, passes the parallelize strategy to fn
        item_wait: Delay between processing individual items within a batch
        iter_wait: Delay between checking completion of submitted batches
        iter: If True, returns an iterator that yields results as they complete
        reduce_fn: Optional function to reduce/combine results from each batch
        worker_queue_len: Number of batches to keep in flight per worker
        **kwargs: Additional keyword args passed to each fn call

    Returns:
        For list-like inputs: A list containing results for each input item
        For dict-like inputs: A dict mapping keys to results
        If iter=True: An iterator yielding results as they become available
        If reduce_fn is provided: The result of applying reduce_fn to all batch results

    Example Usage (parallel map only):
        >>> def process_query_df(query_id, query_df):
        >>>     query_df: pd.DataFrame = set_ranks(query_df, sort_col="example_id")
        >>>     query_df['product_text'] = query_df['product_text'].apply(clean_text)
        >>>     return query_df['product_text'].apply(len).mean()
        >>>
        >>> for mean_query_doc_lens in map_reduce(
        >>>     retrieval_dataset.groupby("query_id"),
        >>>     fn=process_query_df,
        >>>     parallelize='processes',
        >>>     max_workers=20,
        >>>     pbar=dict(miniters=1),
        >>>     batch_size=30,
        >>>     iter=True,
        >>> ):
        >>>     ## Prints the output of each call to process_query_df, which is the mean length of
        >>>     ## product_text for each query_df:
        >>>     print(mean_query_doc_lens)
        >>> 1171.090909090909
        >>> 1317.7931034482758
        >>> 2051.945945945946
        >>> 1249.9375
        >>> ...

    Example Usage (parallel map and reduce):
        >>> def process_query_df(query_id, query_df):
        >>>     query_df: pd.DataFrame = set_ranks(query_df, sort_col="example_id")
        >>>     query_df['product_text'] = query_df['product_text'].apply(clean_text)
        >>>     return query_df['product_text'].apply(len).sum()
        >>>
        >>> def reduce_query_df(l):
        >>>     ## Applied to every batch of outputs from process_query_df
        >>>     ## and then again to thr final list of reduced outputs:
        >>>     return sum(l)
        >>>
        >>> print(map_reduce(
        >>>     retrieval_dataset.groupby("query_id"),
        >>>     fn=process_query_df,
        >>>     parallelize='processes',
        >>>     max_workers=20,
        >>>     pbar=True,
        >>>     batch_size=30,
        >>>     reduce_fn=reduce_query_df,
        >>> ))
        >>> 374453878
    """
    # Convert string parallelization strategy to enum
    parallelize: Parallelize = Parallelize(parallelize)

    # Process batch_size aliases (nrows, chunk_size, etc.)
    Alias.set_num_rows(kwargs)
    batch_size: int = kwargs.get("num_rows", 1)
    Alias.set_progress_bar(kwargs)
    progress_bar: Union[Dict, bool] = kwargs.pop("progress_bar", True)
    if progress_bar is False:
        progress_bar: Optional[Dict] = None
    elif progress_bar is True:
        progress_bar: Optional[Dict] = dict()
    assert progress_bar is None or isinstance(progress_bar, dict)

    # Set appropriate wait times based on execution strategy
    item_wait: float = get_default(
        item_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITEM_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize],
    )
    iter_wait: float = get_default(
        iter_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITER_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize],
    )

    # Forward parallelization strategy if requested
    if forward_parallelize:
        kwargs["parallelize"] = parallelize

    if reduce_fn is not None and iter:
        raise ValueError("Cannot use reduce_fn with iter=True")

    # Functions to process batches
    def process_batch(
        batch_data: List[Any],
        batch_index: int,
        batch_reduce_fn: Optional[Callable],
        **batch_kwargs,
    ):
        """Process a batch of items with optional delay between items"""
        results = []
        for item in batch_data:
            result = fn(*as_tuple(item), **batch_kwargs)
            results.append(result)
            if item_wait > 0:
                time.sleep(item_wait)
        # If using reduce_fn, collect results for later final reduction
        if batch_reduce_fn is not None:
            results = batch_reduce_fn(results)
        return results

    if is_dict_like(struct):
        # Convert to list of (key, value) pairs for batching
        is_dict = True
        struct: List = list(struct.items())
    else:
        is_dict = False

    # For list-like structures or general iterators
    if is_list_or_set_like(struct) or hasattr(struct, "__iter__"):
        # Determine total batches if possible for progress tracking
        total_batches: Optional[int] = None
        if hasattr(struct, "__len__"):
            total_batches: int = math.ceil(len(struct) / batch_size)

        submit_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=total_batches,
            desc="Submitting",
            prefer_kwargs=False,
            unit="item" if batch_size == 1 else "batch",
            disable=(parallelize is Parallelize.sync),
        )
        collect_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=total_batches,
            desc="Processing" if parallelize is Parallelize.sync else "Collecting",
            prefer_kwargs=False,
            unit="item" if batch_size == 1 else "batch",
        )

        # Create iterator to process in batches
        struct_iter: Iterator = builtins.iter(struct)

        # Interspersed submission and collection mode
        def yield_results_interspersed():
            # Create appropriate executor based on parallelization strategy

            executor: Optional[Executor] = dispatch_executor(
                parallelize=parallelize,
                **kwargs,
            )
            try:
                if parallelize in {Parallelize.sync}:
                    max_pending_futures: int = 1
                else:
                    max_pending_futures: int = worker_queue_len * executor._max_workers
                batch_idx = 0
                pending_futures = []
                completed = False
                all_results = []

                while True:
                    # Submit jobs until we reach max_pending_futures limit or run out of items
                    while len(pending_futures) < max_pending_futures and not completed:
                        batch = list(itertools.islice(struct_iter, batch_size))
                        if len(batch) == 0:  # No more items
                            completed = True
                            break

                        # Submit batch for parallel processing
                        fut = dispatch(
                            fn=process_batch,
                            batch_data=batch,
                            batch_index=batch_idx,
                            batch_reduce_fn=reduce_fn,
                            parallelize=parallelize,
                            executor=executor,
                            **filter_kwargs(fn, **kwargs),
                        )
                        pending_futures.append(fut)
                        batch_idx += 1
                        submit_pbar.update(1)

                    if len(pending_futures) == 0 and completed:
                        # We're done - no more pending futures and no more batches
                        break
                    else:
                        # Check for completed futures and yield their results
                        # Get results from completed futures using accumulate_iter
                        for batch_result in accumulate_iter(
                            pending_futures,
                            item_wait=0,  # No additional wait needed since we're checking each iteration
                            iter_wait=0,
                            progress_bar=False,  # We'll update our own progress bar
                        ):
                            collect_pbar.update(1)

                            # Otherwise yield individual results
                            if reduce_fn is not None:
                                all_results.append(batch_result)
                            else:
                                if is_dict:
                                    for k, v in batch_result:
                                        yield k, v
                                else:
                                    for item in batch_result:
                                        yield item
                        pending_futures = []

                # If we have a reduce function, apply it to all collected results
                if reduce_fn is not None:
                    final_result = reduce_fn(all_results)
                    yield final_result

                submit_pbar.success()
                collect_pbar.success()
            finally:
                # Ensure executor is properly cleaned up even if processing fails
                stop_executor(executor)

        # Return the generator that interleaves submission and collection
        gen = yield_results_interspersed()
        if reduce_fn is not None:
            return next(gen)
        if iter:
            return gen
        else:
            return list(gen)
    else:
        raise NotImplementedError(f"Unsupported type: {type_str(struct)}")
