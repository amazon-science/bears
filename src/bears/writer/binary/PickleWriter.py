import io
import pickle
from typing import Any, NoReturn, Optional

from pydantic import constr

from bears.constants import FileContents, FileFormat
from bears.util import FileSystemUtil
from bears.util.aws import S3Util
from bears.writer.Writer import Writer


class PickleWriter(Writer):
    file_contents = [
        FileContents.PICKLED_OBJECT,
    ]
    streams = [io.BytesIO]
    file_formats = [FileFormat.PICKLE]

    def _write_stream(
        self,
        stream: io.IOBase,
        data: Any,
        file_contents: FileContents,
        **kwargs,
    ) -> NoReturn:
        pickle.dump(data, stream)

    def _write_local(
        self,
        local_path: str,
        data: Any,
        file_contents: FileContents,
        file_name: Optional[constr(min_length=1)] = None,
        **kwargs,
    ) -> str:
        if FileSystemUtil.is_path_valid_dir(local_path):
            if file_name is None:
                raise ValueError(f'You must pass `file_name` when writing to local directory "{local_path}".')
            local_path: str = FileSystemUtil.construct_file_path_in_dir(
                path=local_path,
                name=file_name,
                file_ending=self.file_ending,
            )
        FileSystemUtil.put_file_pickle(
            local_path,
            data=data,
            overwrite=True,
        )
        return local_path

    def _write_s3(
        self,
        s3_path: str,
        data: Any,
        file_contents: FileContents,
        **kwargs,
    ) -> str:
        S3Util.put_s3_object_pickle(
            s3_path,
            obj_data=data,
            overwrite=True,
        )
        return s3_path
