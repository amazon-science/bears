import io
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional, Union

from bears.constants import FileContents, MLType, Storage
from bears.document import Document
from bears.reader.Reader import Reader
from bears.util import as_list, retry


class DocumentReader(Reader, ABC):
    file_contents = [
        FileContents.DOCUMENT,
    ]
    streams = [io.BytesIO]
    document_mltype: ClassVar[MLType]

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return super(DocumentReader, cls)._registry_keys() + as_list(cls.document_mltype)

    def _read_document_with_retries(
        self,
        source: Union[str, io.BytesIO],
        storage: Storage,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Document:
        return retry(
            self._read_document,
            retries=self.retry,
            wait=self.retry_wait,
            source=source,
            storage=storage,
            file_contents=file_contents,
            **kwargs,
        )

    def _read_document_multi(
        self,
        sources: List[str],
        storage: Storage,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> List[Document]:
        documents = []
        for source in sources:
            doc = self._read_document_with_retries(
                source=source, storage=storage, file_contents=file_contents, **kwargs
            )
            documents.append(doc)
        return documents

    def _verify_document(self, document: Document) -> Document:
        # Add any document-specific verifications if necessary.
        return document

    # Methods to support reading from different sources:

    def _read_stream(
        self,
        stream: io.BytesIO,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Document:
        return self._verify_document(
            self._read_document_with_retries(
                source=stream, storage=Storage.STREAM, file_contents=file_contents, **kwargs
            )
        )

    def _read_url(
        self,
        url: str,
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Document:
        return self._verify_document(
            self._read_document_with_retries(
                source=url, storage=Storage.URL, file_contents=file_contents, **kwargs
            )
        )

    def _read_local(
        self,
        local_path: Union[str, List[str]],
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Union[Document, List[Document]]:
        if isinstance(local_path, list):
            return [
                self._verify_document(doc)
                for doc in self._read_document_multi(
                    local_path, storage=Storage.LOCAL_FILE_SYSTEM, file_contents=file_contents, **kwargs
                )
            ]
        return self._verify_document(
            self._read_document_with_retries(
                source=local_path, storage=Storage.LOCAL_FILE_SYSTEM, file_contents=file_contents, **kwargs
            )
        )

    def _read_s3(
        self,
        s3_path: Union[str, List[str]],
        file_contents: Optional[FileContents] = None,
        **kwargs,
    ) -> Union[Document, List[Document]]:
        if isinstance(s3_path, list):
            return [
                self._verify_document(doc)
                for doc in self._read_document_multi(
                    s3_path, storage=Storage.S3, file_contents=file_contents, **kwargs
                )
            ]
        return self._verify_document(
            self._read_document_with_retries(
                source=s3_path, storage=Storage.S3, file_contents=file_contents, **kwargs
            )
        )

    @abstractmethod
    def _read_document(self, *args, **kwargs) -> Document:
        pass
