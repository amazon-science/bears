# import io
# from typing import Optional, Union

# import numpy as np
# from pydantic import constr

# from bears.constants import FileContents, FileFormat, Storage
# from bears.reader.asset.audio.AudioReader import AudioReader
# from bears.util import optional_dependency
# from bears.util.aws import S3Util

# with optional_dependency("torchaudio"):

#     class TorchAudioReader(AudioReader):
#         ## Subset of formats supported by imageio:
#         file_formats = [
#             FileFormat.PNG,
#             FileFormat.JPEG,
#             FileFormat.BMP,
#             FileFormat.GIF,
#             FileFormat.ICO,
#             FileFormat.WEBP,
#         ]

#         class Params(AudioReader.Params):
#             mode: constr(min_length=1, max_length=6, strip_whitespace=True) = "RGB"

#         def _read_image(
#             self,
#             source: Union[str, io.BytesIO],
#             storage: Storage,
#             file_contents: Optional[FileContents] = None,
#             postprocess: bool = True,
#             **kwargs,
#         ) -> np.ndarray:
#             if storage is Storage.S3:
#                 source: io.BytesIO = io.BytesIO(S3Util.stream_s3_object(source).read())
#             img: np.ndarray = iio.imread(
#                 source,
#                 **self.params.model_dump(),
#             )
#             if not postprocess:
#                 return img
#             return self._postprocess_image(img, **kwargs)

# class TIFFImageIOReader(ImageIOReader):
#     ## Subset of formats supported by imageio:
#     file_formats = [
#         FileFormat.TIFF,
#     ]

#     class Params(ImageReader.Params):
#         mode: Literal["r"] = "r"  ## In imageio's tifffile plugin, mode is 'r' or 'w'

# def fetch_img_imageio(img_path: str):
#     storage = FileMetadata.detect_storage(img_path)
#     if storage is Storage.LOCAL_FILE_SYSTEM:
#         img_np = iio.imread(
#             img_path,
#             mode="RGB"
#         )
#     elif storage is Storage.S3:
#         img_np = iio.imread(
#             io.BytesIO(S3Util.stream_s3_object(img_path).read()),
#             mode="RGB"
#         )
#     return img_np
#
#
# def np_img_transform(img: np.ndarray, shared_memory=True) -> torch.Tensor:
#     img: np.ndarray = cv2_resize(img)
#     img: torch.Tensor = transform(torch.from_numpy(np.moveaxis(img, -1, 0)))
#     if shared_memory:
#         img: torch.Tensor = img.share_memory_()
#     return img
#
#
# def process_task_data_load_imgs_imageio_concurrent(task_data) -> Dataset:
#     # global task_data_global
#     # task_data_global.append(task_data)
#     try:
#         imgs = task_data.data['First-Image'].apply(
#             lambda img_path: run_concurrent(fetch_img_imageio, img_path)
#         ).apply(accumulate).apply(np_img_transform)
#         if task_data.data.layout is DataLayout.DICT:
#             imgs = imgs.as_torch()
#         task_data.data['img'] = imgs
#         task_data.data_schema.other_schema['img'] = MLType.IMAGE
#     except Exception as e:
#         print(f'Failed for data with ids: {task_data.data["id"].tolist()}')
#         raise e
#     return task_data
