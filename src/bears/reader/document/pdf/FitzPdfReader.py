import io
from typing import Any, Dict, List, Optional, Union

import requests

from bears.constants import FileContents, FileFormat, Storage
from bears.document import Pdf, PdfPage
from bears.FileMetadata import FileMetadata
from bears.util import optional_dependency
from bears.util.aws import S3Util

from .PdfReader import PdfReader

with optional_dependency("fitz"):
    import fitz  # PyMuPDF

    class FitzPdfReader(PdfReader):
        file_formats = [
            FileFormat.PDF,
        ]

        class Params(PdfReader.Params):
            read_images: bool = True
            read_text: bool = True

        def _read_pdf(
            self,
            source: Union[str, io.BytesIO],
            storage: Storage,
            file_contents: Optional[FileContents] = None,
            **kwargs: Any,
        ) -> Pdf:
            if storage is Storage.S3:
                source = io.BytesIO(S3Util.stream_s3_object(source).read())
                doc: fitz.Document = fitz.open(stream=source.read(), filetype="pdf")
            elif storage is Storage.URL:
                response = requests.get(source)
                if not response.ok:
                    raise ValueError(f"Failed to fetch url: '{source}'")
                source_bytes = io.BytesIO(response.content)
                doc: fitz.Document = fitz.open(stream=source_bytes.read(), filetype="pdf")
            elif storage is Storage.LOCAL_FILE_SYSTEM:
                doc: fitz.Document = fitz.open(source)
            else:
                raise NotImplementedError(f"Storage type {storage} is not supported.")
            metadata: Dict = doc.metadata
            pages: List[PdfPage] = []
            for page in doc:
                page_text = None
                if self.params.read_text:
                    page_text = page.get_text()

                page_images = None
                if self.params.read_images:
                    page_images = []
                    for img in page.get_images(full=True):
                        xref = img[0]
                        pix = page.get_pixmap(xref=xref)
                        page_images.append(pix.tobytes("png"))
                pages.append(PdfPage(text=page_text, images=page_images))
            doc.close()

            pdf = Pdf(
                path=FileMetadata.of(source) if storage in {Storage.S3, Storage.LOCAL_FILE_SYSTEM} else None,
                data=pages,
                metadata=metadata,
            )
            return pdf
