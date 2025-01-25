import io
from typing import Callable, Dict, List, NoReturn, Optional, Union

import pandas as pd
from pydantic import constr, model_validator

from bears.constants import QUOTING_MAP, DataLayout, FileFormat, Storage
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

        @model_validator(mode="before")
        @classmethod
        def set_params(cls, params: Dict) -> Dict:
            cls.set_default_param_values(params)
            quoting = params.get("quoting")
            if quoting is not None and quoting not in QUOTING_MAP:
                raise ValueError(f'`quoting` must be in {list(QUOTING_MAP.keys())}; found "{quoting}"')
            params["quoting"] = QUOTING_MAP[quoting]
            return params

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
