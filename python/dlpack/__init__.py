from typing import Any

import torch
import numpy as np
import tensorflow as tf

from .dlpack import (
    write_tensor_to_buffer,
    read_tensor_from_buffer,
    py_buffer,
    Buffer,
    MutableBuffer,
)

__all__ = [
    "Buffer",
    "MutableBuffer",
    "py_buffer",
    "serialize_dlpack",
    "deserialize_dlpack",
    "bytes_to_numpy",
    "bytes_to_torch",
    "bytes_to_tensorflow",
]


def serialize_dlpack(dlpack: Any) -> MutableBuffer:
    if hasattr(dlpack, "__dlpack__"):
        dlpack = dlpack.__dlpack__()
    buf = py_buffer(b"")
    write_tensor_to_buffer(dlpack, buf)
    return buf


def deserialize_dlpack(buf: MutableBuffer) -> Any:
    return read_tensor_from_buffer(buf)


def bytes_to_numpy(buf: MutableBuffer) -> Any:
    return np.from_dlpack(_FakeArr(deserialize_dlpack(buf)))


def bytes_to_torch(buf: MutableBuffer) -> Any:
    return torch.from_dlpack(deserialize_dlpack(buf))


class _FakeArr:
    def __init__(self, pycapsule: Any) -> None:
        self.pycapsule = pycapsule

    def __dlpack__(self, stream=None) -> Any:
        return self.pycapsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


def bytes_to_tensorflow(buf: MutableBuffer) -> Any:
    return tf.experimental.dlpack.from_dlpack(deserialize_dlpack(buf))
