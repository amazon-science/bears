from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
)

import pandas as pd
from autoenum import AutoEnum, auto
from bears.util import is_null
from pydantic import model_validator

from bears.constants import MLType
from bears.processor import SingleColumnProcessor


class NumericImputationStrategy(AutoEnum):
    MEAN = auto()
    MEDIAN = auto()
    MODE = auto()
    MIN = auto()
    MAX = auto()
    CONSTANT = auto()


class NumericMissingValueImputation(SingleColumnProcessor):
    """
    This calculates or fills in the value to be filled in place of nan based on strategy passed as input.
    Params:
    - FILL_VALUE: the value to be filled in when it encounters a NaN (This must be only passed when CONSTANT is strategy)
    - STRATEGY: this indicates what strategy must be used when NaN is encountered
        - MEAN: The "average" you're used to, where you add up all the numbers and then divide by the number of numbers
        - MEDIAN: The "median" is the "middle" value in the list of numbers
        - MODE: The number which appears most often in a set of numbers
        - MIN: The minimum value of the series
        - MAX: The Maximum value of the series
        - CONSTANT: This allows the user to pass in a fill value where that fill value will be imputed
    """

    input_mltypes = [MLType.INT, MLType.FLOAT]
    output_mltype = MLType.FLOAT
    IMPUTE_FN_MAP: ClassVar[Dict[NumericImputationStrategy, Callable]] = {
        NumericImputationStrategy.MODE: lambda _data: _data.mode(dropna=True).compute().iloc[0],
        NumericImputationStrategy.MEAN: lambda _data: _data.mean(skipna=True),
        NumericImputationStrategy.MEDIAN: lambda _data: _data.median(skipna=True),
        NumericImputationStrategy.MIN: lambda _data: _data.min(skipna=True),
        NumericImputationStrategy.MAX: lambda _data: _data.max(skipna=True),
    }

    class Params(SingleColumnProcessor.Params):
        strategy: NumericImputationStrategy
        fill_value: Optional[Any] = None

    imputed_value: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def set_imputed_value(cls, params: Dict) -> Dict:
        cls.set_default_param_values(params)
        if params["params"].strategy is NumericImputationStrategy.CONSTANT:
            if params["params"].fill_value is None:
                raise ValueError(
                    f"Cannot have empty `fill_value` when `strategy` is {NumericImputationStrategy.CONSTANT}"
                )
            params["imputed_value"] = params["params"].fill_value
        elif params["params"].fill_value is not None:
            raise ValueError(
                f"`fill_value` can only be passed when strategy={NumericImputationStrategy.CONSTANT}"
            )
        return params

    def _fit_series(self, data: pd.Series):
        if self.params.strategy is not NumericImputationStrategy.CONSTANT:
            if self.imputed_value is not None:
                raise self.AlreadyFitError
            self.imputed_value: Any = self.IMPUTE_FN_MAP[self.params.strategy](data)

    def transform_single(self, data: Optional[Any]) -> Any:
        if self.imputed_value is None and self.params.strategy is not NumericImputationStrategy.CONSTANT:
            raise self.FitBeforeTransformError
        if is_null(data):
            data = self.imputed_value
        return data
