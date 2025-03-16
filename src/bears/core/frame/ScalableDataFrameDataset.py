from typing import Dict

import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import functional as F

from bears.constants import DataLayout
from bears.core.frame.ScalableDataFrame import ScalableDataFrame
from bears.util import accumulate, run_concurrent


class ScalableDataFrameDataset(IterableDataset):
    """
    An iterable PyTorch dataset implementation for ScalableDataFrame that supports streaming data in batches.

    This dataset enables efficient processing of large dataframes by streaming data and optionally fetching
    assets (like images) on-the-fly. It provides integration with PyTorch's data loading utilities while
    handling the specifics of ScalableDataFrame.

    Note: Currently only supports single-process loading. Multi-worker support is planned for future releases.

    Example usage:
        >>> from torch.utils.data import DataLoader
        >>> import pandas as pd
        >>> from bears.constants import DataLayout
        >>> from bears.core.frame import ScalableDataFrame
        >>>
        >>> # Create a sample dataframe with image paths
        >>> df_small = pd.DataFrame({
        >>>     "img": ["path/to/image1.jpg", "path/to/image2.jpg"],
        >>>     "label": [0, 1]
        >>> })
        >>>
        >>> # Create the dataset
        >>> dataset = ScalableDataFrameDataset(
        >>>     ScalableDataFrame.of(df_small),
        >>>     batch_size=100,
        >>>     schema={"img": "image", "label": "int"},  ## Schema defines column types
        >>>     layout=DataLayout.DICT,  ## Choose between DICT or LIST_OF_DICT
        >>>     fetch_assets=True,  ## Set to True to load images automatically
        >>>     shuffle=True,  ## Whether to shuffle data during iteration
        >>> )
        >>>
        >>> # Create a PyTorch DataLoader (note: batch_size=None as batching is handled by the dataset)
        >>> data_loader = DataLoader(
        >>>     dataset,
        >>>     max_workers=0,  ## Must be 0 until multi-worker support is added
        >>>     batch_size=None,
        >>> )
        >>>
        >>> # Iterate through batches
        >>> for batch_i, batch_sdf in enumerate(data_loader):
        >>>     images = batch_sdf._data["img"]  ## Access loaded image tensors
        >>>     labels = batch_sdf._data["label"]
    """

    def __init__(
        self,
        sdf: ScalableDataFrame,
        schema: Dict,
        batch_size: int,
        layout: DataLayout,
        fetch_assets: bool,
        shuffle: bool,
        ## TODO: add dict of pytorch transforms for each column.
        ## Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms
    ):
        self.__sdf: ScalableDataFrame = sdf
        self.__sdf_len: int = len(sdf)
        self.__schema: Dict = schema
        self.__batch_size: int = batch_size
        self.__layout: DataLayout = layout
        self.__fetch_assets: bool = fetch_assets
        self.__shuffle: bool = shuffle

    @classmethod
    def read_from_disk(cls, img_path: str):
        with Image.open(img_path) as img_f:
            return F.pil_to_tensor(img_f)

    def fetch_data_and_yield(self):
        for sdf_batch in self.__sdf.stream(
            stream_as=self.__layout,
            num_rows=self.__batch_size,
            shuffle=self.__shuffle,
        ):
            if self.__fetch_assets:
                if sdf_batch.layout == DataLayout.DICT:
                    imgs_tensor = torch.stack(
                        accumulate(
                            [
                                run_concurrent(self.read_from_disk, img_path=sdf_batch._data["img"][i])
                                for i in range(len(sdf_batch))
                            ]
                        )
                    )
                    sdf_batch._data["img"] = imgs_tensor
                    sdf_batch.loc[:, "img"] = imgs_tensor
                else:
                    raise NotImplementedError()
            yield sdf_batch

    def __iter__(self):
        ## Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ## In a worker process
            ## TODO: implement multi-worker sdf processing.
            ## Refs:
            ## - https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
            ## -
            raise NotImplementedError("Cannot handle multi-process sdf loading yet")
        ## Single-process sdf loading, return the full iterator
        return self.fetch_data_and_yield()
