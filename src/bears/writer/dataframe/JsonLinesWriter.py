import io
from typing import Callable, NoReturn, Optional, Union

import pandas as pd
from pydantic import constr

from bears.constants import DataLayout, FileFormat, Storage
from bears.core.frame.ScalableDataFrame import ScalableDataFrame
from bears.writer.dataframe.DataFrameWriter import DataFrameWriter


class JsonLinesWriter(DataFrameWriter):
    aliases = ["JsonLinesDataFrameWriter"]  ## Backward compatibility
    file_formats = [FileFormat.JSONLINES]
    dask_multiple_write_file_suffix = ".part"  ## github.com/dask/dask/issues/9044

    class Params(DataFrameWriter.Params):
        orient: constr(min_length=1) = "records"
        lines: bool = True
        index: bool = True

    def _write_sdf(
        self,
        destination: Union[io.IOBase, str],
        sdf: ScalableDataFrame,
        storage: Storage,
        **kwargs,
    ) -> NoReturn:
        sdf.to_json(
            destination,
            **self.filtered_params(pd.DataFrame.to_json),
        )

    def _write_dask_sdf(
        self,
        destination: Union[io.IOBase, str],
        sdf: ScalableDataFrame,
        storage: Storage,
        is_dir: bool,
        name_function: Optional[Callable[[int], str]] = None,
        **kwargs,
    ) -> NoReturn:
        from dask.dataframe.io.json import to_json as Dask_to_json

        if storage is Storage.STREAM:
            ## Convert dask dataframe to Pandas and write to stream:
            self._write_sdf(
                destination=destination, sdf=sdf.as_layout(DataLayout.PANDAS), storage=storage, **kwargs
            )
        elif not is_dir:
            ## Dask's to_json always writes a folder, even if you want it to write a single file.
            ## As a result, we must convert to Pandas to write a single file:
            self._write_sdf(
                destination=destination, sdf=sdf.as_layout(DataLayout.PANDAS), storage=storage, **kwargs
            )
        else:
            ## We are writing multiple files to a directory (either in local or remote).
            assert name_function is not None, "We require a `name_function` when writing to a directory."
            sdf.to_json(
                destination,
                name_function=name_function,
                ## Dask .to_json params: docs.dask.org/en/stable/generated/dask.dataframe.to_json.html
                **self.filtered_params(Dask_to_json, pd.DataFrame.to_json),
            )
