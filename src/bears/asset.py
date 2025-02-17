from abc import ABC
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Union

import numpy as np
from pydantic import conint, model_validator

from bears.constants import (
    AVAILABLE_TENSOR_TYPES,
    SHORTHAND_TO_TENSOR_LAYOUT_MAP,
    TENSOR_LAYOUT_TO_SHORTHAND_MAP,
    DataLayout,
    MLType,
)
from bears.FileMetadata import FileMetadata
from bears.util import Parameters, Registry, String, optional_dependency, type_str


class Asset(Parameters, Registry, ABC):
    _allow_subclass_override = True

    mltype: ClassVar[MLType]
    path: Optional[FileMetadata] = None
    data: Any
    layout: DataLayout

    @model_validator(mode="before")
    @classmethod
    def validate_params(cls, params: Dict) -> Dict:
        params["layout"]: DataLayout = cls.detect_layout(params["data"])
        return params

    @classmethod
    def detect_layout(cls, data: Any, raise_error: bool = True) -> Optional[DataLayout]:
        for layout, dtype in AVAILABLE_TENSOR_TYPES.items():
            if isinstance(data, dtype):
                return layout
        if raise_error:
            raise ValueError(f"Cannot detect layout for data of type {type_str(data)}.")
        return None

    @classmethod
    def _registry_keys(cls) -> MLType:
        return cls.mltype

    def as_tensor(self, tensor_type_or_layout: Union[DataLayout, str], **kwargs) -> Optional[Any]:
        tensor_layout: DataLayout = tensor_type_or_layout
        if not isinstance(tensor_layout, DataLayout):
            if DataLayout.matches_any(tensor_layout):
                tensor_layout: DataLayout = DataLayout.from_str(tensor_layout)
            else:
                tensor_layout: DataLayout = SHORTHAND_TO_TENSOR_LAYOUT_MAP[
                    String.str_normalize(tensor_type_or_layout)
                ]
        if tensor_layout not in TENSOR_LAYOUT_TO_SHORTHAND_MAP:
            raise ValueError(
                f"Argument `tensor_type_or_layout`: {tensor_layout} is not a valid tensor layout. "
                f"supported tensor layouts: {list(TENSOR_LAYOUT_TO_SHORTHAND_MAP.values())}"
            )
        if tensor_layout not in AVAILABLE_TENSOR_TYPES:
            raise ValueError(
                f"Corresponding package has not been installed for argument `tensor_type_or_layout`: {tensor_layout}`; "
                f"available packages: {list(AVAILABLE_TENSOR_TYPES.keys())}"
            )
        if tensor_layout is DataLayout.NUMPY:
            return self.numpy(**kwargs)
        if tensor_layout is DataLayout.TORCH:
            return self.torch(**kwargs)
        raise NotImplementedError(f"Unsupported value of `tensor_type_or_layout`: {tensor_type_or_layout}")

    def numpy(self, error: Literal["raise", "warn", "ignore"] = "raise", **kwargs) -> Optional[Any]:
        if self.layout is DataLayout.NUMPY:
            return self.data
        if self.layout is DataLayout.TORCH:
            return self.data.cpu().numpy()
        if error == "raise":
            pass
        return None

    def torch(self, error: Literal["raise", "warn", "ignore"] = "raise", **kwargs) -> Optional[Any]:
        if self.layout is DataLayout.NUMPY:
            import torch

            return torch.from_numpy(self.data)
        if self.layout is DataLayout.TORCH:
            return self.data
        if error == "raise":
            pass
        return None


class Image(Asset):
    mltype = MLType.IMAGE

    height: conint(ge=1)
    width: conint(ge=1)
    color_mode: Literal["G", "RGB", "BRG"]
    channels: Literal["first", "last"]

    def to_pil_image(self) -> Optional[Any]:
        img: np.ndarray = self.to_channels_last().numpy()
        with optional_dependency("torchvision", "PIL"):
            from PIL import Image as PILImage
            from torchvision.transforms.functional import to_pil_image

            img: PILImage = to_pil_image(img)
            return img
        return None

    def to_channels_first(self) -> Asset:
        if self.channels == "first":
            return self
        if self.layout is DataLayout.NUMPY:
            moveaxis: Callable = np.moveaxis
        elif self.layout is DataLayout.TORCH:
            import torch

            moveaxis: Callable = torch.moveaxis
        else:
            raise NotImplementedError()
        img = moveaxis(self.data, -1, 0)
        return Image(
            data=img,
            channels="first",
            **self.model_dump(exclude={"data", "channels"}),
        )

    def to_channels_last(self) -> Asset:
        if self.channels == "last":
            return self
        if self.layout is DataLayout.NUMPY:
            moveaxis: Callable = np.moveaxis
        elif self.layout is DataLayout.TORCH:
            import torch

            moveaxis: Callable = torch.moveaxis
        else:
            raise NotImplementedError()
        img = moveaxis(self.data, 0, -1)
        return Image(
            data=img,
            channels="last",
            **self.model_dump(exclude={"data", "channels"}),
        )


class Audio(Asset):
    mltype = MLType.AUDIO

    sampling_rate: conint(ge=1)
