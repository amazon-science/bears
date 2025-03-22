import functools
import inspect
import io
import json
import math
import pprint
import random
import re
import string
import types
from ast import literal_eval
from collections import defaultdict
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union, KeysView, ValuesView

import numpy as np
import pandas as pd
from pydantic import confloat, conint, validate_call

from ._function import get_fn_spec, is_function
from ._import import np_bool, np_floating, np_integer, optional_dependency
from ._string_data import RANDOM_ADJECTIVES, RANDOM_NAME_LEFT, RANDOM_NAME_RIGHT, RANDOM_NOUNS, RANDOM_VERBS

StructuredBlob = Union[List, Dict, List[Dict]]  ## used for type hints.
KERNEL_START_DT: datetime = datetime.now()

_PUNCTUATION_REMOVAL_TABLE = str.maketrans(
    "",
    "",
    string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_SPACE = str.maketrans(
    "",
    "",
    " " + string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    " " + string.punctuation,  ## Will be removed
)

_PUNCTUATION_REMOVAL_TABLE_WITH_NUMBERS = str.maketrans(
    "",
    "",
    "1234567890" + string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_NUMBERS = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "1234567890" + string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_SPACE_AND_NUMBERS = str.maketrans(
    "",
    "",
    "1234567890 " + string.punctuation,  ## Will be removed
)
_PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE_AND_NUMBERS = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "1234567890 " + string.punctuation,  ## Will be removed
)


class NeverFailJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # print(f'Running NeverFailJsonEncoder')
        if isinstance(obj, (np_integer, int)):
            return int(obj)
        elif isinstance(obj, (np_bool, bool)):
            return bool(obj)
        elif isinstance(obj, (np_floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, list, set, tuple)):
            return obj.tolist()
        elif isinstance(obj, complex):
            return obj.real, obj.imag
        elif isinstance(
            obj,
            (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.LambdaType,
                functools.partial,
            ),
        ):
            return {"<function>": f"{obj.__module__}.{obj.__qualname__}{inspect.signature(obj)}"}
        with optional_dependency("torch"):
            import torch

            if isinstance(obj, torch.dtype):
                return str(obj)
        try:
            return super(NeverFailJsonEncoder, self).default(obj)
        except TypeError:
            obj_members: List[str] = []
            for k, v in obj.__dict__.items():
                if is_function(v):
                    continue
                k_str: str = str(k)
                v_str: str = "..."
                obj_members.append(f"{k_str}={v_str}")
            obj_members_str: str = ", ".join(obj_members)
            return f"{obj.__class__.__name__}({obj_members_str})"


## Taken from: https://github.com/django/django/blob/master/django/utils/baseconv.py#L101
class BaseConverter:
    decimal_digits: str = "0123456789"

    def __init__(self, digits, sign="-"):
        self.sign = sign
        self.digits = digits
        if sign in self.digits:
            raise ValueError("Sign character found in converter base digits.")

    def __repr__(self):
        return "<%s: base%s (%s)>" % (self.__class__.__name__, len(self.digits), self.digits)

    def encode(self, i):
        neg, value = self.convert(i, self.decimal_digits, self.digits, "-")
        if neg:
            return self.sign + value
        return value

    def decode(self, s):
        neg, value = self.convert(s, self.digits, self.decimal_digits, self.sign)
        if neg:
            value = "-" + value
        return int(value)

    def convert(self, number, from_digits, to_digits, sign):
        if str(number)[0] == sign:
            number = str(number)[1:]
            neg = 1
        else:
            neg = 0

        # make an integer out of the number
        x = 0
        for digit in str(number):
            x = x * len(from_digits) + from_digits.index(digit)

        # create the result in base 'len(to_digits)'
        if x == 0:
            res = to_digits[0]
        else:
            res = ""
            while x > 0:
                digit = x % len(to_digits)
                res = to_digits[digit] + res
                x = int(x // len(to_digits))
        return neg, res


class String:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    EMPTY: str = ""
    SPACE: str = " "
    DOUBLE_SPACE: str = SPACE * 2
    FOUR_SPACE: str = SPACE * 4
    TAB: str = "\t"
    NEWLINE: str = "\n"
    WINDOWS_NEWLINE: str = "\r"
    BACKSLASH: str = "\\"
    SLASH: str = "/"
    PIPE: str = "|"
    SINGLE_QUOTE: str = "'"
    DOUBLE_QUOTE: str = '"'
    COMMA: str = ","
    COMMA_SPACE: str = ", "
    COMMA_NEWLINE: str = ",\n"
    HYPHEN: str = "-"
    DOUBLE_HYPHEN: str = "--"
    DOT: str = "."
    ASTERISK: str = "*"
    DOUBLE_ASTERISK: str = "**"
    QUESTION_MARK: str = "?"
    CARET: str = "^"
    DOLLAR: str = "$"
    UNDERSCORE: str = "_"
    COLON: str = ":"
    SEMICOLON: str = ";"
    EQUALS: str = "="
    LEFT_PAREN: str = "("
    RIGHT_PAREN: str = ")"
    BACKTICK: str = "`"
    TILDE: str = "~"

    MATCH_ALL_REGEX_SINGLE_LINE: str = CARET + DOT + ASTERISK + DOLLAR
    MATCH_ALL_REGEX_MULTI_LINE: str = DOT + ASTERISK

    S3_PREFIX: str = "s3://"
    FILE_PREFIX: str = "file://"
    HTTP_PREFIX: str = "http://"
    HTTPS_PREFIX: str = "https://"
    PORT_REGEX: str = ":(\d+)"
    DOCKER_REGEX: str = "\d+\.dkr\.ecr\..*.amazonaws\.com/.*"

    DEFAULT_CHUNK_NAME_PREFIX: str = "part"

    FILES_TO_IGNORE: str = ["_SUCCESS", ".DS_Store"]

    UTF_8: str = "utf-8"

    FILE_SIZE_UNITS: Tuple[str, ...] = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    ## FILE_SIZE_REGEX taken from: https://rgxdb.com/r/4IG91ZFE
    ## Matches: "2", "2.5", "2.5b", "2.5B", "2.5k", "2.5K", "2.5kb", "2.5Kb", "2.5KB", "2.5kib", "2.5KiB", "2.5kiB"
    ## Does not match: "2.", "2ki", "2ib", "2.5KIB"
    FILE_SIZE_REGEX = r"^(\d*\.?\d+)((?=[KMGTkgmt])([KMGTkgmt])(?:i?[Bb])?|[Bb]?)$"

    ALPHABET: Tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    ALPHABET_CAPS: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    ALPHABET_CAPS_NO_DIGITS: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    RANDOM_NAME_LEFT: List[str] = RANDOM_NAME_LEFT
    RANDOM_NAME_RIGHT: List[str] = RANDOM_NAME_RIGHT
    RANDOM_ADJECTIVES: List[str] = RANDOM_ADJECTIVES
    RANDOM_NOUNS: List[str] = RANDOM_NOUNS
    RANDOM_VERBS: List[str] = RANDOM_VERBS

    BASE2_ALPHABET: str = "01"
    BASE16_ALPHABET: str = "0123456789ABCDEF"
    BASE56_ALPHABET: str = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz"
    BASE36_ALPHABET: str = "0123456789abcdefghijklmnopqrstuvwxyz"
    BASE62_ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    BASE64_ALPHABET: str = BASE62_ALPHABET + "-_"

    BASE_CONVERTER_MAP: Dict[int, BaseConverter] = {
        2: BaseConverter(BASE2_ALPHABET),
        16: BaseConverter(BASE16_ALPHABET),
        36: BaseConverter(BASE36_ALPHABET),
        56: BaseConverter(BASE56_ALPHABET),
        62: BaseConverter(BASE62_ALPHABET),
        64: BaseConverter(BASE64_ALPHABET, sign="$"),
    }

    @classmethod
    def str_normalize(
        cls, x: str, *, remove: Optional[Union[str, Tuple, List, Set]] = (" ", "-", "_")
    ) -> str:
        ## Found to be faster than .translate() and re.sub() on Python 3.10.6
        if remove is None:
            remove: Set[str] = set()
        if isinstance(remove, str):
            remove: Set[str] = set(remove)
        assert isinstance(remove, (list, tuple, set))
        if len(remove) == 0:
            return str(x).lower()
        out: str = str(x)
        for rem in set(remove).intersection(set(out)):
            out: str = out.replace(rem, "")
        out: str = out.lower()
        return out

    @classmethod
    def punct_normalize(
        cls, x: str, *, lowercase: bool = True, space: bool = True, numbers: bool = False
    ) -> str:
        punct_table = {
            (False, False, False): _PUNCTUATION_REMOVAL_TABLE,
            (True, False, False): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE,
            (False, True, False): _PUNCTUATION_REMOVAL_TABLE_WITH_SPACE,
            (True, True, False): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE,
            (False, False, True): _PUNCTUATION_REMOVAL_TABLE_WITH_NUMBERS,
            (True, False, True): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_NUMBERS,
            (False, True, True): _PUNCTUATION_REMOVAL_TABLE_WITH_SPACE_AND_NUMBERS,
            (True, True, True): _PUNCTUATION_REMOVAL_TABLE_WITH_LOWERCASE_AND_SPACE_AND_NUMBERS,
        }[(lowercase, space, numbers)]
        return str(x).translate(punct_table)

    @classmethod
    def whitespace_normalize(cls, text: str, remove_newlines: bool = False):
        ## Remove trailing whitespace at the end of each line
        text: str = re.sub(r"\s+$", "", text, flags=re.MULTILINE)

        if remove_newlines:
            text: str = text.replace("\n", "")
        else:
            ## Replace double newlines with single newlines
            text: str = re.sub(r"\n\n+", "\n", text)

        ## Replace double spaces with single spaces
        text: str = re.sub(r"  +", " ", text)
        return text.strip()

    @classmethod
    def format_exception_msg(cls, ex: Exception, short: bool = False, prefix: str = "[ERROR]") -> str:
        ## Ref: https://stackoverflow.com/a/64212552
        tb = ex.__traceback__
        trace = []
        while tb is not None:
            trace.append(
                {
                    "filename": tb.tb_frame.f_code.co_filename,
                    "function_name": tb.tb_frame.f_code.co_name,
                    "lineno": tb.tb_lineno,
                }
            )
            tb = tb.tb_next
        out = f'{prefix}: {type(ex).__name__}: "{str(ex)}"'
        if short:
            out += "\nTrace: "
            for trace_line in trace:
                out += f"{trace_line['filename']}#{trace_line['lineno']}; "
        else:
            out += "\nTraceback:"
            for trace_line in trace:
                out += f"\n\t{trace_line['filename']} line {trace_line['lineno']}, in {trace_line['function_name']}..."
        return out.strip()

    @classmethod
    def str_format_args(cls, x: str, named_only: bool = True) -> List[str]:
        ## Ref: https://stackoverflow.com/a/46161774/4900327
        args: List[str] = [str(tup[1]) for tup in string.Formatter().parse(x) if tup[1] is not None]
        if named_only:
            args: List[str] = [arg for arg in args if not arg.isdigit() and len(arg) > 0]
        return args

    @classmethod
    def assert_not_empty_and_strip(cls, string: str, error_message: str = "") -> str:
        cls.assert_not_empty(string, error_message)
        return string.strip()

    @classmethod
    def strip_if_not_empty(cls, string: str) -> str:
        if cls.is_not_empty(string):
            return string.strip()
        return string

    @classmethod
    def is_not_empty(cls, string: str) -> bool:
        return isinstance(string, str) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_bytes(cls, string: bytes) -> bool:
        return isinstance(string, bytes) and len(string.strip()) > 0

    @classmethod
    def is_not_empty_str_or_bytes(cls, string: Union[str, bytes]) -> bool:
        return cls.is_not_empty(string) or cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty(cls, string: Any) -> bool:
        return not cls.is_not_empty(string)

    @classmethod
    def is_empty_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_bytes(string)

    @classmethod
    def is_empty_str_or_bytes(cls, string: Any) -> bool:
        return not cls.is_not_empty_str_or_bytes(string)

    @classmethod
    def assert_not_empty(cls, string: Any, error_message: str = ""):
        assert cls.is_not_empty(string), error_message

    @classmethod
    def assert_not_empty_bytes(cls, string: Any, error_message: str = ""):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def assert_not_empty_str_or_bytes(cls, string: Any, error_message: str = ""):
        assert cls.is_not_empty_str_or_bytes(string), error_message

    @classmethod
    def is_int(cls, string: Any) -> bool:
        """
        Checks if an input string is an integer.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '-123' but False for '123.0', '1.23', '-1.23' and '1e2'
        """
        try:
            int(string)
            return True
        except Exception:
            return False

    @classmethod
    def is_float(cls, string: Any) -> bool:
        """
        Checks if an input string is a floating-point value.
        :param string: input string
        :raises: error when input is not a string
        :return: True for '123', '1.23', '123.0', '-123', '-123.0', '1e2', '1.23e-5', 'NAN' & 'nan'; but False for 'abc'
        """
        try:
            float(string)  ## Will return True for NaNs as well.
            return True
        except Exception:
            return False

    @classmethod
    def is_prefix(cls, prefix: str, strings: Union[List[str], Set[str]]) -> bool:
        cls.assert_not_empty(prefix)
        if isinstance(strings, str):
            strings = [strings]
        return True in {string.startswith(prefix) for string in strings}

    @classmethod
    def remove_prefix(cls, string: str, prefix: str) -> str:
        cls.assert_not_empty(prefix)
        if string.startswith(prefix):
            string = string[len(prefix) :]
        return string

    @classmethod
    def remove_suffix(cls, string: str, suffix: str) -> str:
        cls.assert_not_empty(suffix)
        if string.endswith(suffix):
            string = string[: -len(suffix)]
        return string

    @classmethod
    def join_human(
        cls,
        l: Union[List, Tuple, Set],
        sep: str = ",",
        final_join: str = "and",
        oxford_comma: bool = False,
    ) -> str:
        l: List = list(l)
        if len(l) == 1:
            return str(l[0])
        out: str = ""
        for x in l[:-1]:
            out += " " + str(x) + sep
        if not oxford_comma:
            out: str = cls.remove_suffix(out, sep)
        x = l[-1]
        out += f" {final_join} " + str(x)
        return out.strip()

    @classmethod
    def convert_str_to_type(cls, val: str, expected_type: Type) -> Any:
        assert isinstance(expected_type, type)
        if isinstance(val, expected_type):
            return val
        if expected_type is str:
            return str(val)
        if expected_type is bool and isinstance(val, str):
            val = val.lower().strip().capitalize()  ## literal_eval does not parse "false", only "False".
        out = literal_eval(String.assert_not_empty_and_strip(str(val)))
        if expected_type is float and isinstance(out, int):
            out = float(out)
        if expected_type is int and isinstance(out, float) and int(out) == out:
            out = int(out)
        if expected_type is tuple and isinstance(out, list):
            out = tuple(out)
        if expected_type is list and isinstance(out, tuple):
            out = list(out)
        if expected_type is set and isinstance(out, (list, tuple)):
            out = set(out)
        if expected_type is bool and out in [0, 1]:
            out = bool(out)
        if type(out) is not expected_type:
            raise ValueError(f"Input value {val} cannot be converted to {str(expected_type)}")
        return out

    @classmethod
    def readable_bytes(cls, size_in_bytes: int, decimals: int = 3) -> str:
        sizes: Dict[str, float] = cls.convert_size_from_bytes(size_in_bytes, unit=None, decimals=decimals)
        sorted_sizes: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(sizes.items(), key=lambda item: item[1])
        ]
        size_unit, size_val = None, None
        for size_unit, size_val in sorted_sizes:
            if size_val >= 1:
                break
        return f"{size_val} {size_unit}"

    @classmethod
    def convert_size_from_bytes(
        cls,
        size_in_bytes: int,
        unit: Optional[str] = None,
        decimals: int = 3,
    ) -> Union[Dict, float]:
        size_in_bytes = float(size_in_bytes)
        cur_size = size_in_bytes
        sizes = {}
        if size_in_bytes == 0:
            for size_name in cls.FILE_SIZE_UNITS:
                sizes[size_name] = 0.0
        else:
            for size_name in cls.FILE_SIZE_UNITS:
                val: float = round(cur_size, decimals)
                i = 1
                while val == 0:
                    val = round(cur_size, decimals + i)
                    i += 1
                sizes[size_name] = val
                i = int(math.floor(math.log(cur_size, 1024)))
                cur_size = cur_size / 1024
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.upper()
            assert unit in cls.FILE_SIZE_UNITS
            return sizes[unit]
        return sizes

    @classmethod
    def convert_size_to_bytes(cls, size_in_human_readable: str) -> int:
        size_in_human_readable: str = cls.assert_not_empty_and_strip(size_in_human_readable).upper()
        size_selection_regex = f"""(\d+(?:\.\d+)?) *({cls.PIPE.join(cls.FILE_SIZE_UNITS)})"""  ## This uses a non-capturing group: https://stackoverflow.com/a/3512530/4900327
        matches = re.findall(size_selection_regex, size_in_human_readable)
        if len(matches) != 1 or len(matches[0]) != 2:
            raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')
        val, unit = matches[0]
        val = float(val)
        for file_size_unit in cls.FILE_SIZE_UNITS:
            if unit == file_size_unit:
                return int(round(val))
            val = val * 1024
        raise ValueError(f'Cannot convert value "{size_in_human_readable}" to bytes.')

    @classmethod
    def readable_seconds(
        cls,
        time_in_seconds: Union[float, timedelta],
        *,
        decimals: int = 2,
        short: bool = False,
    ) -> str:
        if isinstance(time_in_seconds, timedelta):
            time_in_seconds: float = time_in_seconds.total_seconds()
        times: Dict[str, float] = cls.convert_time_from_seconds(
            time_in_seconds,
            unit=None,
            decimals=decimals,
            short=short,
        )
        sorted_times: List[Tuple[str, float]] = [
            (k, v) for k, v in sorted(times.items(), key=lambda item: item[1])
        ]
        time_unit, time_val = None, None
        for time_unit, time_val in sorted_times:
            if time_val >= 1:
                break
        if decimals <= 0:
            time_val = int(time_val)
        if short:
            return f"{time_val}{time_unit}"
        return f"{time_val} {time_unit}"

    @classmethod
    def convert_time_from_seconds(
        cls,
        time_in_seconds: float,
        unit: Optional[str] = None,
        decimals: int = 3,
        short: bool = False,
    ) -> Union[Dict, float]:
        TIME_UNITS = {
            "nanoseconds": 1e-9,
            "microseconds": 1e-6,
            "milliseconds": 1e-3,
            "seconds": 1.0,
            "mins": 60,
            "hours": 60 * 60,
            "days": 24 * 60 * 60,
        }
        if short:
            TIME_UNITS = {
                "ns": 1e-9,
                "us": 1e-6,
                "ms": 1e-3,
                "s": 1.0,
                "min": 60,
                "hr": 60 * 60,
                "d": 24 * 60 * 60,
            }
        time_in_seconds = float(time_in_seconds)
        times: Dict[str, float] = {
            time_unit: round(time_in_seconds / TIME_UNITS[time_unit], decimals) for time_unit in TIME_UNITS
        }
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in TIME_UNITS
            return times[unit]
        return times

    @classmethod
    def readable_number(
        cls,
        n: Union[float, int],
        decimals: int = 3,
        short: bool = True,
        scientific: bool = False,
    ) -> str:
        if n == 0:
            return "0"
        assert abs(n) > 0
        if 0 < abs(n) < 1:
            scientific: bool = True
        if scientific:
            n_unit: str = ""
            n_val: str = f"{n:.{decimals}e}"
        else:
            numbers: Dict[str, float] = cls.convert_number(
                abs(n),
                unit=None,
                decimals=decimals,
                short=short,
            )
            sorted_numbers: List[Tuple[str, float]] = [
                (k, v) for k, v in sorted(numbers.items(), key=lambda item: item[1])
            ]
            n_unit: Optional[str] = None
            n_val: Optional[float] = None
            for n_unit, n_val in sorted_numbers:
                if n_val >= 1:
                    break
            if decimals <= 0:
                n_val: int = int(n_val)
            if n_val == int(n_val):
                n_val: int = int(n_val)
        if n < 0:
            n_val: str = f"-{n_val}"
        if short:
            return f"{n_val}{n_unit}".strip()
        return f"{n_val} {n_unit}".strip()

    @classmethod
    def convert_number(
        cls,
        n: float,
        unit: Optional[str] = None,
        decimals: int = 3,
        short: bool = False,
    ) -> Union[Dict, float]:
        assert n >= 0
        N_UNITS = {
            "": 1e0,
            "thousand": 1e3,
            "million": 1e6,
            "billion": 1e9,
            "trillion": 1e12,
            "quadrillion": 1e15,
            "quintillion": 1e18,
        }
        if short:
            N_UNITS = {
                "": 1e0,
                "K": 1e3,
                "M": 1e6,
                "B": 1e9,
                "T": 1e12,
                "Qa": 1e15,
                "Qi": 1e18,
            }
        n: float = float(n)
        numbers: Dict[str, float] = {n_unit: round(n / N_UNITS[n_unit], decimals) for n_unit in N_UNITS}
        if unit is not None:
            assert isinstance(unit, str)
            unit = unit.lower()
            assert unit in N_UNITS
            return numbers[unit]
        return numbers

    @classmethod
    def detect_case(
        cls,
        s: str,
    ) -> Literal["snake", "screaming_snake", "kebab", "train", "camel", "pascal", "studly"]:
        """
        Detects the case of the input literal name.

        Returns:
            One of:
                - 'snake' for snake_case,
                - 'screaming_snake' for SCREAMING_SNAKE_CASE,
                - 'kebab' for kebab-case,
                - 'train' for TRAIN-CASE,
                - 'camel' for camelCase,
                - 'pascal' for PascalCase.
                - 'studly' for StUDlyCaSe

        Raises:
            ValueError: If the input string is empty, contains unsupported characters,
                        or uses mixed delimiters.
        """
        if len(s.strip()) == 0:
            raise ValueError("Empty string cannot be detected as a standard case for literal names.")

        # Early check: ensure string contains only allowed characters.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", s):
            raise ValueError(
                f"String '{s}' contains unsupported characters in a literal name. "
                f"Only letters, digits, underscores, and hyphens are allowed."
            )
        if "-" in s and "_" in s:
            raise ValueError(
                f"String '{s}' contains mixed delimiters and is not a standard case format for literal names."
            )

        # If underscores are present, assume snake_case or screaming_snake.
        elif "_" in s:
            # Determine if the string is all uppercase (screaming snake) or not.
            if s.upper() == s:
                return "screaming_snake"  ## SCREAMING_SNAKE_CASE
            elif s.lower() == s:
                return "snake"  ## snake_case
            raise ValueError(
                f"String '{s}' contains both underscores and a combination of uppercase "
                f"and lowercase, which is not a standard case format for literal names."
            )

        # If hyphens are present, assume kebab-case.
        elif "-" in s:
            # Determine if the string is all uppercase (screaming snake) or not.
            if s.upper() == s:
                return "train"  ## TRAIN-CASE
            elif s.lower() == s:
                return "kebab"  ## kebab-case
            raise ValueError(
                f"String '{s}' contains both hyphens and a combination of uppercase "
                f"and lowercase, which is not a standard case format for literal names."
            )
        else:
            # For strings without delimiters, assume camelCase or PascalCase.
            if s[0].islower():
                return "camel"  ## camelCase
            elif s[0].isupper():
                return "pascal"  ## PascalCase
            else:
                return "studly"  ## StUDlyCaSe, sTuDlYCaSE, etc

        raise ValueError(f"String '{s}' does not match any of the supported case formats for literal names.")

    @classmethod
    def convert_case(
        cls,
        s: str,
        target_case: Literal[
            "upper",
            "lower",
            "snake",
            "screaming_snake",
            "kebab",
            "train",
            "camel",
            "pascal",
            "studly",
        ],
    ) -> list:
        """
        Convert a string from one case format to another.

        This function detects the source case format automatically and converts
        the string to the specified target case format.

        Supported case formats:
            - 'snake': snake_case (words separated by underscores, all lowercase)
            - 'screaming_snake': SCREAMING_SNAKE_CASE (words separated by underscores, all uppercase)
            - 'kebab': kebab-case (words separated by hyphens, all lowercase)
            - 'train': TRAIN-CASE (words separated by hyphens, all uppercase)
            - 'camel': camelCase (no separators, first word lowercase, subsequent words capitalized)
            - 'pascal': PascalCase (no separators, all words capitalized)

        Args:
            s: The input string to convert
            target_case: The target case format to convert to

        Returns:
            The converted string in the target case format

        Raises:
            ValueError: If the source case cannot be detected or if the target case is unsupported

        Examples:
            >>> assert String.convert_case("CachedResultsStep", "kebab") == "cached-results-step"
            >>> assert String.convert_case("cachedResultsStep", "snake") == "cached_results_step"
            >>> assert String.convert_case("cached_results_step", "pascal") == "CachedResultsStep"
            >>> assert String.convert_case("cached-results-step", "camel") == "cachedResultsStep"
            >>> assert String.convert_case("CACHED_RESULTS_STEP", "camel") == "cachedResultsStep"
            >>> assert String.convert_case("Test123Case", "snake") == "test_123_case"
            >>> assert String.convert_case("another_test_case", "kebab") == "another-test-case"
        """
        target_case: str = (
            target_case.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
            .removesuffix("_")
            .removesuffix("case")
            .removesuffix("_")
        )
        if target_case == "upper":
            return s.upper()
        elif target_case == "lower":
            return s.lower()
        source_case: str = cls.detect_case(s)
        if source_case in ("snake", "screaming_snake"):
            words: List[str] = s.split("_")
        elif source_case in {"kebab", "train"}:
            words: List[str] = s.split("-")
        elif source_case in ("camel", "pascal"):
            # Regex handles:
            #  - Acronyms (e.g. "HTML"),
            #  - Normal words,
            #  - Numbers.
            words: List[str] = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", s)
        elif source_case == "studly":
            raise ValueError(
                "Due to its lack of standardization, it is not possible to convert from studly case yet."
            )
        else:
            raise ValueError(f"Unsupported input case: '{source_case}'")

        if target_case == "snake":
            return "_".join(word.lower() for word in words)
        elif target_case == "screaming_snake":
            return "_".join(word.upper() for word in words)
        elif target_case == "kebab":
            return "-".join(word.lower() for word in words)
        elif target_case == "train":
            return "-".join(word.lower() for word in words)
        elif target_case == "camel":
            if len(words) == 0:
                return ""
            return words[0].lower() + "".join(word.capitalize() for word in words[1:])
        elif target_case == "pascal":
            return "".join(word.capitalize() for word in words)
        elif target_case == "studly":
            return "".join(c.lower() if random.random() <= 0.5 else c.upper() for c in s)
        else:
            raise ValueError(f"Unsupported target case: '{target_case}'")

    @classmethod
    def jsonify(
        cls,
        blob: StructuredBlob,
        *,
        minify: bool = False,
    ) -> str:
        if minify:
            return json.dumps(blob, indent=None, separators=(cls.COMMA, cls.COLON), cls=NeverFailJsonEncoder)
        else:
            return json.dumps(blob, cls=NeverFailJsonEncoder, indent=4)

    @classmethod
    def get_num_zeros_to_pad(cls, max_i: int) -> int:
        assert isinstance(max_i, int) and max_i >= 1
        num_zeros = math.ceil(math.log10(max_i))  ## Ref: https://stackoverflow.com/a/51837162/4900327
        if max_i == 10**num_zeros:  ## If it is a power of 10
            num_zeros += 1
        return num_zeros

    @classmethod
    def pad_zeros(cls, i: int, max_i: int = int(1e12)) -> str:
        assert isinstance(i, int)
        assert i >= 0
        assert isinstance(max_i, int)
        assert max_i >= i, f"Expected max_i to be >= current i; found max_i={max_i}, i={i}"
        num_zeros: int = cls.get_num_zeros_to_pad(max_i)
        return f"{i:0{num_zeros}}"

    @classmethod
    def stringify(
        cls,
        d: Union[Dict, List, Tuple, Set, Any],
        *,
        sep: str = ",",
        key_val_sep: str = "=",
        literal: bool = False,
        nested_literal: bool = True,
    ) -> str:
        if isinstance(d, (dict, defaultdict)):
            if nested_literal:
                out: str = sep.join(
                    [
                        f"{k}"
                        f"{key_val_sep}"
                        f"{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=True, nested_literal=True)}"
                        for k, v in sorted(list(d.items()), key=lambda x: x[0])
                    ]
                )
            else:
                out: str = sep.join(
                    [
                        f"{k}"
                        f"{key_val_sep}"
                        f"{cls.stringify(v, sep=sep, key_val_sep=key_val_sep, literal=False, nested_literal=False)}"
                        for k, v in sorted(list(d.items()), key=lambda x: x[0])
                    ]
                )
        elif isinstance(d, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
            try:
                s = sorted(list(d))
            except TypeError:  ## Sorting fails
                s = list(d)
            out: str = sep.join(
                [
                    f"{cls.stringify(x, sep=sep, key_val_sep=key_val_sep, literal=nested_literal, nested_literal=nested_literal)}"
                    for x in s
                ]
            )
        else:
            out: str = repr(d)
        if literal:
            if isinstance(d, list):
                out: str = f"[{out}]"
            elif isinstance(d, np.ndarray):
                out: str = f"np.array([{out}])"
            elif isinstance(d, pd.Series):
                out: str = f"pd.Series([{out}])"
            elif isinstance(d, tuple):
                if len(d) == 1:
                    out: str = f"({out},)"
                else:
                    out: str = f"({out})"
            elif isinstance(d, (set, frozenset)):
                out: str = f"({out})"
            elif isinstance(d, (dict, defaultdict)):
                out: str = f"dict({out})"
        return out

    @classmethod
    def destringify(cls, s: str) -> Any:
        if isinstance(s, str):
            try:
                val = literal_eval(s)
            except ValueError:
                val = s
        else:
            val = s
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return val
        return val

    @classmethod
    @validate_call
    def random(
        cls,
        shape: Tuple = (1,),
        length: Union[conint(ge=1), Tuple[conint(ge=1), conint(ge=1)]] = 6,
        spaces_prob: Optional[confloat(ge=0.0, le=1.0)] = None,
        alphabet: Tuple = ALPHABET,
        seed: Optional[int] = None,
        unique: bool = False,
    ) -> Union[str, np.ndarray]:
        if isinstance(length, int):
            min_num_chars: int = length
            max_num_chars: int = length
        else:
            min_num_chars, max_num_chars = length
        assert min_num_chars <= max_num_chars, (
            f"Must have min_num_chars ({min_num_chars}) <= max_num_chars ({max_num_chars})"
        )
        if spaces_prob is not None:
            num_spaces_to_add: int = int(round(len(alphabet) * spaces_prob / (1 - spaces_prob), 0))
            alphabet = alphabet + num_spaces_to_add * (cls.SPACE,)

        ## Ref: https://stackoverflow.com/a/25965461/4900327
        np_random = np.random.RandomState(seed=seed)
        random_alphabet_lists = np_random.choice(alphabet, shape + (max_num_chars,))
        random_strings: np.ndarray = np.apply_along_axis(
            arr=random_alphabet_lists,
            func1d=lambda random_alphabet_list: "".join(random_alphabet_list)[
                : np_random.randint(min_num_chars, max_num_chars + 1)
            ],
            axis=len(shape),
        )
        if shape == (1,):
            return random_strings[0]
        if unique:
            random_strings_flatten1d: np.ndarray = random_strings.ravel()
            if len(set(random_strings_flatten1d)) != len(random_strings_flatten1d):
                ## Call it recursively:
                random_strings: np.ndarray = cls.random(
                    shape=shape,
                    length=length,
                    spaces_prob=spaces_prob,
                    alphabet=alphabet,
                    seed=seed,
                    unique=unique,
                )
        return random_strings

    @classmethod
    def random_name(
        cls,
        count: int = 1,
        *,
        sep: str = HYPHEN,
        order: Tuple[str, ...] = ("adjective", "verb", "noun"),
        seed: Optional[int] = None,
    ) -> Union[List[str], str]:
        cartesian_product_parts: List[List[str]] = []
        assert len(order) > 0
        for order_part in order:
            if order_part == "verb":
                cartesian_product_parts.append(cls.RANDOM_VERBS)
            elif order_part == "adjective":
                cartesian_product_parts.append(cls.RANDOM_ADJECTIVES)
            elif order_part == "noun":
                cartesian_product_parts.append(cls.RANDOM_NOUNS)
            else:
                raise NotImplementedError(f'Unrecognized part of the order sequence: "{order_part}"')

        out: List[str] = [
            sep.join(parts)
            for parts in cls.__random_cartesian_product(*cartesian_product_parts, seed=seed, n=count)
        ]
        if count == 1:
            return out[0]
        return out

    @staticmethod
    def __random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
        rnd = random.Random(seed)
        cartesian_idxs: Set[Tuple[int, ...]] = set()
        list_lens: List[int] = [len(l) for l in lists]
        max_count: int = 1
        for l_len in list_lens:
            max_count *= l_len
        if max_count < n:
            raise ValueError(f"At most {max_count} cartesian product elements can be created.")
        while len(cartesian_idxs) < n:
            rnd_idx: Tuple[int, ...] = tuple(rnd.randint(0, l_len - 1) for l_len in list_lens)
            if rnd_idx not in cartesian_idxs:
                cartesian_idxs.add(rnd_idx)
                elem = []
                for l_idx, l in zip(rnd_idx, lists):
                    elem.append(l[l_idx])
                yield elem

    @classmethod
    def parse_datetime(cls, dt: Union[str, int, float, datetime]) -> datetime:
        if isinstance(dt, datetime):
            return dt
        elif type(dt) in [int, float]:
            return datetime.fromtimestamp(dt)
        elif isinstance(dt, str):
            return datetime.fromisoformat(dt)
        raise NotImplementedError(f"Cannot parse datetime from value {dt} with type {type(dt)}")

    @classmethod
    def now(cls, **kwargs) -> str:
        dt: datetime = datetime.now()
        return cls.readable_datetime(dt, **kwargs)

    @classmethod
    def kernel_start_time(cls, **kwargs) -> str:
        return cls.readable_datetime(KERNEL_START_DT, **kwargs)

    @classmethod
    def readable_datetime(
        cls,
        dt: datetime,
        *,
        human: bool = False,
        microsec: bool = True,
        tz: bool = True,
        **kwargs,
    ) -> str:
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if human:
            format_str: str = "%d%b%Y-%H:%M:%S"
            microsec: bool = False
        else:
            format_str: str = "%Y-%m-%dT%H:%M:%S"
        if microsec:
            format_str += ".%f"
        split_tz_colon: bool = False
        if tz and dt.tzinfo is not None:
            if human:
                format_str += "+%Z"
            else:
                format_str += "%z"
                split_tz_colon: bool = True
        out: str = dt.strftime(format_str).strip()
        if split_tz_colon:  ## Makes the output exactly like dt.isoformat()
            out: str = out[:-2] + ":" + out[-2:]
        return out

    @classmethod
    def convert_integer_to_base_n_str(cls, integer: int, base: int) -> str:
        assert isinstance(integer, int)
        assert isinstance(base, int) and base in cls.BASE_CONVERTER_MAP, (
            f"Param `base` must be an integer in {list(cls.BASE_CONVERTER_MAP.keys())}; found: {base}"
        )
        return cls.BASE_CONVERTER_MAP[base].encode(integer)

    @classmethod
    def hash(cls, val: Union[str, int, float, List, Dict], max_len: int = 256, base: int = 62) -> str:
        """
        Constructs a hash of a JSON object or value.
        :param val: any valid JSON value (including str, int, float, list, and dict).
        :param max_len: the maximum length of the output hash (will truncate upto this length).
        :param base: the base of the output hash.
            Defaults to base56, which encodes the output in a ASCII-chars
        :return: SHA256 hash.
        """

        def hash_rec(val, base):
            if isinstance(val, (set, frozenset, KeysView)):
                val: List = sorted(list(val))
            if isinstance(val, (list, tuple, ValuesView)):
                return hash_rec(",".join([hash_rec(x, base=base) for x in val]), base=base)
            elif isinstance(val, dict):
                return hash_rec(
                    [
                        f"{hash_rec(k, base=base)}:{hash_rec(v, base=base)}"
                        for k, v in sorted(val.items(), key=lambda kv: kv[0])
                    ],
                    base=base,
                )
            elif is_function(val):
                return hash_rec(get_fn_spec(val).source_body)
            return cls.convert_integer_to_base_n_str(
                int(sha256(str(val).encode("utf8")).hexdigest(), 16), base=base
            )

        return hash_rec(val, base)[:max_len]

    @classmethod
    def fuzzy_match(
        cls,
        string: str,
        strings_to_match: Union[str, List[str]],
        replacements: Tuple = (SPACE, HYPHEN, SLASH),
        repl_char: str = UNDERSCORE,
    ) -> Optional[str]:
        """Gets the closest fuzzy-matched string from the list, or else returns None."""
        if not isinstance(strings_to_match, list) and not isinstance(strings_to_match, tuple):
            assert isinstance(strings_to_match, str), (
                f"Input must be of a string or list of strings; found type "
                f"{type(strings_to_match)} with value: {strings_to_match}"
            )
            strings_to_match: List[str] = [strings_to_match]
        string: str = str(string).lower()
        strings_to_match_repl: List[str] = [str(s).lower() for s in strings_to_match]
        for repl in replacements:
            string: str = string.replace(repl, repl_char)
            strings_to_match_repl: List[str] = [s.replace(repl, repl_char) for s in strings_to_match_repl]
        for i, s in enumerate(strings_to_match_repl):
            if string == s:
                return strings_to_match[i]
        return None

    @classmethod
    def is_fuzzy_match(cls, string: str, strings_to_match: List[str]) -> bool:
        """Returns whether or not there is a fuzzy-matched string in the list"""
        return cls.fuzzy_match(string, strings_to_match) is not None

    @classmethod
    def header(cls, text: str, width: int = 65, border: str = "=") -> str:
        out = ""
        out += border * width + cls.NEWLINE
        out += ("{:^" + str(width) + "s}").format(text) + cls.NEWLINE
        out += border * width + cls.NEWLINE
        return out

    @classmethod
    def prefix_each_line(cls, text: str, prefix: str) -> str:
        return re.sub("^", prefix, text, flags=re.MULTILINE)

    @classmethod
    def suffix_each_line(cls, text: str, prefix: str) -> str:
        return re.sub("$", prefix, text, flags=re.MULTILINE)

    @classmethod
    def is_stream(cls, obj) -> bool:
        return isinstance(obj, io.IOBase) and hasattr(obj, "read")

    @classmethod
    def pretty(cls, d: Any, max_width: int = 100) -> str:
        if isinstance(d, dict):
            return pprint.pformat(d, indent=4, width=max_width)
        return pprint.pformat(d, width=max_width)

    @classmethod
    def dedupe(cls, text: str, dedupe: str) -> str:
        while (2 * dedupe) in text:
            text: str = text.replace(2 * dedupe, dedupe)
        return text
