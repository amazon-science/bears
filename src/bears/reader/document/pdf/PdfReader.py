import io
from abc import ABC, abstractmethod
from typing import Optional, Union

from bears.constants import FileContents, Storage, MLType
from bears.document import Pdf
from bears.reader.document.DocumentReader import DocumentReader


class PdfReader(DocumentReader, ABC):
    document_mltype = MLType.PDF

    def _read_document(
        self,
        source: Union[str, io.BytesIO],
        storage: Storage,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Pdf:
        return self._read_pdf(source=source, storage=storage, file_contents=file_contents, **kwargs)

    @abstractmethod
    def _read_pdf(
        self,
        source: Union[str, io.BytesIO],
        storage: Storage,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Pdf:
        pass
