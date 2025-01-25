from typing import NoReturn
from typing import Any, Callable, ClassVar, Dict, Generator, List, Literal, Optional, Set, Tuple, Type, Union
_LIBRARY_NAME: str = 'bears'
import bears.util
import bears.constants
from bears.asset import Asset
from bears.FileMetadata import FileMetadata
from bears.core.frame.ScalableDataFrame import ScalableDataFrame, ScalableOrRaw, \
    ScalableDataFrameRawType, ScalableDataFrameOrRaw, is_scalable
from bears.core.frame.ScalableSeries import ScalableSeries, ScalableSeriesRawType,ScalableSeriesOrRaw
from bears.reader import Reader 
from bears.writer import Writer

to_sdf = ScalableDataFrame.of
to_ss = ScalableSeries.of

def of(data: ScalableOrRaw, **kwargs) -> Union[ScalableDataFrame, ScalableSeries]:
    kwargs['return_ss'] = True
    return ScalableDataFrame.of(data, **kwargs)
