import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Union

import pandas as pd
from pydantic import FilePath, conint, constr

from bears.util.jupyter import JupyterNotebook
from bears.util.language import MutableParameters, String, binary_search, safe_validate_arguments

_DEBUG: str = "DEBUG"
_INFO: str = "INFO"
_WARNING: str = "WARNING"
_ERROR: str = "ERROR"
_FATAL: str = "FATAL"

_LOG_LEVELS: Dict[str, int] = {
    f"{_DEBUG}": logging.DEBUG,
    f"{_INFO}": logging.INFO,
    f"{_WARNING}": logging.WARNING,
    f"{_ERROR}": logging.ERROR,
    f"{_FATAL}": logging.FATAL,
}
_LOG_LEVELS_REVERSE: Dict[int, str] = {
    logging.DEBUG: f"{_DEBUG}",
    logging.INFO: f"{_INFO}",
    logging.WARNING: f"{_WARNING}",
    logging.ERROR: f"{_ERROR}",
    logging.FATAL: f"{_FATAL}",
}
## Add new level names for our purposes to avoid getting logs from other libraries.
for custom_log_level_name, custom_log_level in _LOG_LEVELS.items():
    logging.addLevelName(level=custom_log_level, levelName=custom_log_level_name)


class _Log(MutableParameters):
    _log_level: Literal[
        _DEBUG,
        _INFO,
        _WARNING,
        _ERROR,
        _FATAL,
    ] = _INFO
    _file_log_level: Literal[
        _DEBUG,
        _INFO,
        _WARNING,
        _ERROR,
        _FATAL,
    ] = _DEBUG
    _log_file_path: FilePath = None
    _log_file_logger: Optional[logging.Logger] = None
    _is_jupyter: bool = JupyterNotebook.is_notebook()

    @safe_validate_arguments
    def set_log_file(
        self,
        file_path: FilePath,
        actor_name: Optional[constr(min_length=1, max_length=64)] = None,
    ):
        if self._log_file_logger is not None:
            raise RuntimeError(
                f'Cannot set log file multiple times; already logging to "{self._log_file_path}"'
            )
        if actor_name is not None:
            formatter = logging.Formatter(
                f"[{actor_name} @ %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC%z"
            )
        else:
            formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC%z")
        root_logger: logging.Logger = logging.getLogger()  ## Gets root logger
        root_logger.handlers[:] = []  ## Removes all existing handlers
        file_handler: logging.Handler = logging.FileHandler(file_path, mode="a+")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(_LOG_LEVELS[f"{_DEBUG}"])
        self._log_file_logger = root_logger
        self._log_file_path = file_path

    @safe_validate_arguments
    def set_log_level(self, log_level: Literal[_DEBUG, _INFO, _WARNING, _ERROR, _FATAL]):
        log_level: str = String.assert_not_empty_and_strip(log_level).upper()
        self._log_level = log_level

    @safe_validate_arguments
    def set_file_log_level(self, log_level: Literal[_DEBUG, _INFO, _WARNING, _ERROR, _FATAL]):
        log_level: str = String.assert_not_empty_and_strip(log_level).upper()
        self._file_log_level = log_level

    def log(self, *data, level: Union[str, int, float], flush: bool = False, **kwargs):
        if isinstance(level, (int, float)):
            ## Translate to our log level:
            level: str = _LOG_LEVELS_REVERSE[
                binary_search(
                    list(_LOG_LEVELS_REVERSE.keys()),
                    target=level,
                    return_tuple=True,
                )[0]
            ]  ## E.g. level=23 returns (_DEBUG=20, WARN=30), we should pick _DEBUG (lower of the two).
        data_str: str = " ".join([self.to_log_str(x) for x in data])
        ## print at the appropriate level:
        if _LOG_LEVELS[self._log_level] <= _LOG_LEVELS[level]:
            ## Logs to both stdout and file logger if setup:
            if self._is_jupyter:
                from IPython.display import display

                for x in data:
                    if isinstance(x, (pd.DataFrame, pd.Series)):
                        display(x)
                    else:
                        print(self.to_log_str(x), end="", flush=flush)
                print("", flush=flush)
            else:
                print(data_str, flush=flush)

        if self._log_file_logger is not None and _LOG_LEVELS[self._file_log_level] <= _LOG_LEVELS[level]:
            self._log_file_logger.log(
                ## We log to file at debug level:
                level=_LOG_LEVELS[f"{_DEBUG}"],
                msg=data_str,
            )

    def debug(self, *data, **kwargs):
        self.log(*data, level=f"{_DEBUG}", **kwargs)

    def info(self, *data, **kwargs):
        self.log(*data, level=f"{_INFO}", **kwargs)

    def warning(self, *data, **kwargs):
        self.log(*data, level=f"{_WARNING}", **kwargs)

    def error(self, *data, **kwargs):
        self.log(*data, level=f"{_ERROR}", **kwargs)

    def fatal(self, *data, **kwargs):
        self.log(*data, level=f"{_FATAL}", **kwargs)

    @classmethod
    def to_log_str(cls, data: Any, *, df_num_rows: conint(ge=1) = 10) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return "\n" + String.jsonify(data)
        if isinstance(data, (list, set, frozenset, tuple)):
            return "\n" + String.pretty(data, max_width=int(1e6))
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) <= df_num_rows:
                return "\n" + str(data.to_markdown())
            return (
                "\n"
                + str(data.head(df_num_rows // 2).to_markdown())
                + f"\n...({len(data) - df_num_rows} more rows)...\n"
                + str(data.tail(df_num_rows // 2).to_markdown())
            )
        return String.pretty(data, max_width=int(1e6))


Log: _Log = _Log()  ## Creates a singleton


@contextmanager
def ignore_warnings():
    pd_chained_assignment: Optional[str] = pd.options.mode.chained_assignment  # default='warn'
    with warnings.catch_warnings():  ## Ref: https://stackoverflow.com/a/14463362
        warnings.simplefilter("ignore")
        ## Stops Pandas SettingWithCopyWarning in output. Ref: https://stackoverflow.com/a/20627316
        pd.options.mode.chained_assignment = None
        yield
    pd.options.mode.chained_assignment = pd_chained_assignment


@contextmanager
def ignore_stdout():
    devnull = open(os.devnull, "w")
    stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = stdout


@contextmanager
def ignore_stderr():
    devnull = open(os.devnull, "w")
    stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = stderr


@contextmanager
def ignore_stdout_and_stderr():
    with ignore_stdout():
        with ignore_stderr():
            yield


@contextmanager
def ignore_warnings_and_stdout():
    with ignore_warnings():
        with ignore_stdout():
            with ignore_stderr():
                yield


@contextmanager
def ignore_logging(disable_upto: int = logging.CRITICAL):
    prev_disable_level: int = logging.root.manager.disable
    logging.disable(disable_upto + 1)
    try:
        yield
    finally:
        logging.disable(prev_disable_level)


@contextmanager
def ignore_all_output():
    with ignore_stdout():
        with ignore_warnings():
            with ignore_stderr():
                with ignore_logging():
                    yield


@contextmanager
def ignore_nothing():
    yield
