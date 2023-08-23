from typing import Any

import torch
import numpy as np
import tensorflow as tf

from .dlpack import (
    write_tensor_to_buffer,
    read_tensor_from_buffer,
    py_buffer,
    PyBuffer,
    MutableBuffer,
)

__all__ = [
    "PyBuffer",
    "MutableBuffer",
    "py_buffer",
    "serialize_dlpack",
    "deserialize_dlpack",
    "bytes_to_numpy",
    "bytes_to_torch",
    "bytes_to_tensorflow",
]


class _FakeArr:
    def __init__(self, pycapsule: Any) -> None:
        self.pycapsule = pycapsule

    def __dlpack__(self, stream=None) -> Any:
        return self.pycapsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


def serialize_dlpack(dlpack: Any) -> PyBuffer:
    if hasattr(dlpack, "__dlpack__"):
        dlpack = dlpack.__dlpack__()
    return write_tensor_to_buffer(dlpack)


def deserialize_dlpack(buf: bytes) -> Any:
    return read_tensor_from_buffer(py_buffer(buf))


def bytes_to_numpy(buf: bytes) -> Any:
    return np.from_dlpack(_FakeArr(deserialize_dlpack(buf)))


def bytes_to_torch(buf: bytes) -> Any:
    return torch.from_dlpack(deserialize_dlpack(buf))


def bytes_to_tensorflow(buf: MutableBuffer) -> Any:
    return tf.experimental.dlpack.from_dlpack(deserialize_dlpack(buf))
