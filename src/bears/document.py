from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional

from bears.constants import MLType
from bears.FileMetadata import FileMetadata
from bears.util import Parameters, Registry


class Document(Parameters, Registry, ABC):
    mltype: ClassVar[MLType]
    path: Optional[FileMetadata] = None
    data: Any


class PdfPage(Parameters):
    metadata: Dict = {}
    text: Optional[str]
    images: Optional[List[bytes]]


class Pdf(Document):
    mltype = MLType.PDF
    metadata: Dict = {}

    @property
    def pages(self) -> List[PdfPage]:
        return self.data
