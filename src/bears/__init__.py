_LIBRARY_NAME: str = "bears"
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Literal,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from bears.util import *
from bears.constants import *
from bears.asset import *
from bears.document import *
from bears.FileMetadata import *
from bears.core import *
from bears.reader import *
from bears.writer import *
from bears.processor import *

to_sdf = ScalableDataFrame.of
to_ss = ScalableSeries.of

def of(data: ScalableOrRaw, **kwargs) -> Union[ScalableDataFrame, ScalableSeries]:
    kwargs["return_series"] = True
    return ScalableDataFrame.of(data, **kwargs)
