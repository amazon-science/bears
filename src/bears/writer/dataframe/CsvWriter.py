import io
from typing import *

import pandas as pd
from pydantic import constr, validator

from bears.constants import DataLayout, FileFormat, Storage
from bears.core.frame.ScalableDataFrame import ScalableDataFrame
from bears.util import String
from bears.writer.dataframe.DataFrameWriter import DataFrameWriter


class CsvWriter(DataFrameWriter):
    aliases = ["CsvDataFrameWriter"]  ## Backward compatibility
    file_formats = [FileFormat.CSV]
    dask_multiple_write_file_suffix = ".part"  ## github.com/dask/dask/issues/9044

    class Params(DataFrameWriter.Params):
        sep: constr(min_length=1, max_length=3) = String.COMMA
        header: Union[bool, List[str]] = True
        quoting: Optional[str] = None
        index: Optional[int] = None

        @validator("quoting")
        def validate_quoting(cls, quoting):
            if quoting is None:
                return None
            if quoting not in cls.QUOTING_MAP:
                raise ValueError(f'`quoting` must be in {list(cls.QUOTING_MAP.keys())}; found "{quoting}"')
            return cls.QUOTING_MAP[quoting]

    def _write_sdf(
        self,
        destination: Union[io.IOBase, str],
        sdf: ScalableDataFrame,
        storage: Storage,
        **kwargs,
    ) -> NoReturn:
        return sdf.to_csv(
            destination,
            **self.filtered_params(pd.DataFrame.to_csv),
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
        from dask.dataframe.io.csv import to_csv as Dask_to_csv

        if storage is Storage.STREAM:
            ## Convert dask dataframe to Pandas and write to stream:
            self._write_sdf(destination, sdf=sdf.as_layout(DataLayout.PANDAS), storage=storage, **kwargs)
        elif not is_dir:
            ## We want to write a single file:
            sdf.to_csv(
                destination,
                single_file=True,
                ## Dask .to_csv params: docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_csv.html
                **self.filtered_params(Dask_to_csv, pd.DataFrame.to_csv),
            )
        else:
            ## We are writing multiple files to a directory (either in local or remote).
            assert name_function is not None, "We require a `name_function` when writing to a directory."
            sdf.to_csv(
                destination,
                name_function=name_function,  ## This writes output files as .csv.part: github.com/dask/dask/issues/9044
                ## Dask .to_csv params: docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_csv.html
                **self.filtered_params(Dask_to_csv, pd.DataFrame.to_csv),
            )


class TsvWriter(CsvWriter):
    aliases = ["TsvDataFrameWriter"]  ## Backward compatibility
    file_formats = [FileFormat.TSV]

    class Params(CsvWriter.Params):
        sep: constr(min_length=1, max_length=3) = String.TAB
