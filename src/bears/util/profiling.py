import math
import time
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from pydantic import confloat, model_validator

from bears.util.language import Alias, MutableParameters, Parameters, String, set_param_from_alias
from bears.util.logging import Log


def measure_time_ms(fn: Callable) -> Tuple[Any, float]:
    start: float = time.perf_counter()
    output: Any = fn()
    end: float = time.perf_counter()
    return output, 1000 * (end - start)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

    pass


class Timer(Parameters):
    task: str
    logger: Optional[Callable] = Log.info
    silent: bool = False
    single_line: bool = False  ## Single-line printing
    i: Optional[int] = None
    max_i: Optional[int] = None
    _start_dt: Optional[datetime] = None
    _start_time_ns: Optional[int] = None
    _end_dt: Optional[datetime] = None
    _end_time_ns: Optional[int] = None

    def __init__(self, task: str = "", **kwargs):
        super(Timer, self).__init__(task=task, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _set_timer_params(cls, params: Dict) -> Dict:
        set_param_from_alias(params, param="logger", alias=["log"])
        Alias.set_silent(params, default=False)
        if "logger" in params and params["logger"] in {None, False}:
            params["logger"]: Optional[Callable] = None
        return params

    @property
    def has_started(self) -> bool:
        return self._start_time_ns is not None

    @property
    def has_stopped(self) -> bool:
        return self._end_time_ns is not None

    def time_taken(self, format: str) -> Union[timedelta, int, float, str]:
        if format in {str, "str", "string"}:
            return self.time_taken_str
        elif format in {"s", "sec", "seconds"}:
            return self.time_taken_sec
        elif format in {"ms", "milli", "millis", "millisec", "milliseconds"}:
            return self.time_taken_ms
        elif format in {"us", "micro", "micros", "microsec", "microseconds"}:
            return self.time_taken_us
        elif format in {"ns", "nano", "nanos", "nanosec", "nanoseconds"}:
            return self.time_taken_ns
        elif format in {"dt", "td", "datetime", "timedelta"}:
            return self.time_taken_td
        raise NotImplementedError(f"Unsupported `format` with type {type(format)} and value: {format}")

    @property
    def start_datetime(self) -> datetime:
        return self._start_dt

    @property
    def end_datetime(self) -> datetime:
        return self._end_dt

    @property
    def start_time_str(self) -> str:
        return String.readable_datetime(self._start_dt)

    @property
    def end_time_str(self) -> str:
        return String.readable_datetime(self._end_dt)

    @property
    def time_taken_str(self) -> str:
        return String.readable_seconds(self.time_taken_sec, decimals=2)

    @property
    def time_taken_human(self) -> str:
        return String.readable_seconds(self.time_taken_sec, decimals=2)

    @property
    def time_taken_sec(self) -> float:
        return self.time_taken_ns / 1e9

    @property
    def time_taken_ms(self) -> float:
        return self.time_taken_ns / 1e6

    @property
    def time_taken_us(self) -> float:
        return self.time_taken_ns / 1e3

    @property
    def time_taken_ns(self) -> int:
        self._check_started()
        if self.has_stopped:
            return self._end_time_ns - self._start_time_ns
        return time.perf_counter_ns() - self._start_time_ns

    @property
    def time_taken_td(self) -> timedelta:
        ## Python timedelta does not have nanosecond resolution: https://github.com/python/cpython/issues/59648
        return timedelta(microseconds=self.time_taken_us)

    def _check_started(self):
        if not self.has_started:
            raise TimerError("Timer has not been started. Use .start() to start it.")

    def _check_not_started(self):
        if self.has_started:
            raise TimerError(f"Timer has already been started at {String.readable_datetime(self._start_dt)}")

    def _check_stopped(self):
        if not self.has_stopped:
            raise TimerError("Timer has not been stopped. Use .stop() to stop it.")

    def _check_not_stopped(self):
        if self.has_stopped:
            raise TimerError(f"Timer has already been stopped at {String.readable_datetime(self._end_dt)}")

    def start(self):
        self._check_not_started()
        self._start_time_ns = time.perf_counter_ns()
        now: datetime = datetime.now()
        now: datetime = now.replace(tzinfo=now.astimezone().tzinfo)
        self._start_dt = now
        if self.should_log and not self.single_line:
            self.logger(self._start_msg())

    def alert(self, text: Optional[str] = None):
        self._check_started()
        self._check_not_stopped()
        if self.should_log:
            self.logger(self._alert_msg(text))

    def stop(self):
        self._check_not_stopped()
        self._end_time_ns: int = time.perf_counter_ns()
        now: datetime = datetime.now()
        now: datetime = now.replace(tzinfo=now.astimezone().tzinfo)
        self._end_dt = now
        if self.should_log:
            self.logger(self._end_msg())

    @property
    def should_log(self) -> bool:
        return self.logger is not None and self.silent is False

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer, report elapsed time."""
        self.stop()

    def _start_msg(self) -> str:
        out: str = ""
        out += self._task_msg()
        out += self._idx_msg()
        out += f"Started at {String.readable_datetime(self._start_dt)}..."
        return out

    def _alert_msg(self, text: Optional[str] = None) -> str:
        out: str = ""
        out += self._task_msg()
        out += self._idx_msg()
        out += f"Timer has been running for {String.readable_seconds(self.time_taken_sec, decimals=2)}."
        if isinstance(text, str):
            out += f" {text}"
        return out

    def _end_msg(self) -> str:
        out: str = ""
        out += self._task_msg()
        out += self._idx_msg()
        if self.single_line:
            out += (
                f"Started at {String.readable_datetime(self._start_dt)}, "
                f"completed in {String.readable_seconds(self.time_taken_sec, decimals=2)}."
            )
            return out
        out += f"...completed in {String.readable_seconds(self.time_taken_sec, decimals=2)}."
        return out

    def _task_msg(self) -> str:
        out: str = ""
        if len(self.task) > 0:
            out += f"({self.task}) "
        return out

    def _idx_msg(self) -> str:
        out: str = ""
        if self.i is not None and self.max_i is not None:
            out += (
                f"[{String.pad_zeros(i=self.i + 1, max_i=self.max_i)}/"
                f"{String.pad_zeros(i=self.max_i, max_i=self.max_i)}] "
            )
        elif self.i is not None:
            out += f"[{self.i}] "
        return out


class Timeout(MutableParameters):
    timeout: confloat(gt=0)  ## In seconds.
    last_used_time: float = time.time()

    @property
    def has_expired(self) -> bool:
        return self.last_used_time + self.timeout < time.time()

    def reset_timeout(self):
        self.last_used_time: float = time.time()


class Timeout1Min(Timeout):
    timeout: confloat(gt=0, le=60)


class Timeout15Min(Timeout):
    timeout: confloat(gt=0, le=60 * 15)


class Timeout1Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60)


class Timeout24Hr(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24)


class TimeoutNever(Timeout):
    timeout: float = math.inf


class Timeout1Week(Timeout):
    timeout: confloat(gt=0, le=60 * 60 * 24 * 7)
