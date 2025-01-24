from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Tuple, Type, Union

import pandas as pd

from bears.core.frame.ScalableSeries import SS_DEFAULT_NAME, ScalableSeries
from bears.util import get_default, is_function, wrap_fn_output

TensorScalableSeries = "TensorScalableSeries"


class TensorScalableSeries(ScalableSeries, ABC):
    TensorType: ClassVar[Type]

    @property
    @abstractmethod
    def tensor_shape(self) -> Tuple[int,]:
        pass

    def __len__(self):
        if self.is_0d:
            ## 0-dimensional tensor, e.g. torch.tensor(True) ## Note small "t" in torch.tensor
            return 1
        return self.tensor_shape[0]

    def as_pandas(self, **kwargs) -> pd.Series:
        return pd.Series(self.numpy(**kwargs), name=get_default(self._name, SS_DEFAULT_NAME))

    def _to_frame_raw(self, **kwargs):
        kwargs["name"] = get_default(self._name, SS_DEFAULT_NAME)
        return self.pandas(**kwargs).to_frame(**kwargs)

    @property
    @abstractmethod
    def is_0d(self) -> bool:
        pass

    def __getattr__(self, attr_name: str) -> Union[Any, TensorScalableSeries]:
        """Forwards calls to the respective method of Tensor class."""
        out = super().__getattr__(attr_name)
        if is_function(out):
            return wrap_fn_output(out, wrapper_fn=self._to_scalable)
        if isinstance(out, self.TensorType):
            return self._constructor(out)
        return out

    """
    ---------------------------------------------
    Function application, GroupBy & window
    ---------------------------------------------
    """

    def map(
        self,
        arg: Union[Callable, Dict, ScalableSeries],
        na_action: Optional[Literal["ignore"]] = None,
    ) -> ScalableSeries:
        raise NotImplementedError("Cannot execute .map() over a Tensor series")
