_LIBRARY_NAME: str = 'bears'
import bears.util
import bears.constants
from bears.asset import Asset
from bears.FileMetadata import FileMetadata
from bears.core.frame import ScalableDataFrame, ScalableSeries
from bears.reader import Reader 
from bears.writer import Writer

to_sdf = ScalableDataFrame.of
to_ss = ScalableSeries.of