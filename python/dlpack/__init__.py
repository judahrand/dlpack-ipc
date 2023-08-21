from typing import Any

from .dlpack import dlpack_to_bytes, bytes_to_dlpack

__all__ = [
    "serialize_dlpack",
    "deserialize_dlpack",
    "bytes_to_numpy",
    "bytes_to_torch",
    "bytes_to_tensorflow",
]


def serialize_dlpack(dlpack: Any) -> bytes:
    if hasattr(dlpack, "__dlpack__"):
        dlpack = dlpack.__dlpack__()
    return dlpack_to_bytes(dlpack)


def deserialize_dlpack(buf: bytes) -> Any:
    return bytes_to_dlpack(buf)


def bytes_to_numpy(buf: bytes) -> Any:
    import numpy as np

    return np.from_dlpack(_FakeArr(deserialize_dlpack(buf)))


def bytes_to_torch(buf: bytes) -> Any:
    import torch

    return torch.from_dlpack(deserialize_dlpack(buf))


class _FakeArr:
    def __init__(self, pycapsule: Any) -> None:
        self.pycapsule = pycapsule

    def __dlpack__(self, stream=None) -> Any:
        return self.pycapsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


def bytes_to_tensorflow(buf: bytes) -> Any:
    import tensorflow as tf

    return tf.experimental.dlpack.from_dlpack(deserialize_dlpack(buf))
