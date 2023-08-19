from __future__ import annotations

import enum
import ctypes
import dataclasses
from typing import Any, TYPE_CHECKING
from .v1 import dlpack_pb2

from .ctypes import (
    _c_str_dltensor,
    DLManagedTensor as C_DLManagedTensor,
)

if TYPE_CHECKING:
    import torch
    import tensorflow as tf
    import numpy.typing as npt

PyBUF_READ = 0x100

ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

ctypes.pythonapi.PyMemoryView_FromMemory.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
ctypes.pythonapi.PyMemoryView_FromMemory.restype = ctypes.py_object


@dataclasses.dataclass
class DLPackVersion:
    """Structure representing the version of DLPack."""
    major: int
    minor: int

    def to_proto(self) -> dlpack_pb2.DLPackVersion:
        return dlpack_pb2.DLPackVersion(
            major=self.major,
            minor=self.minor,
        )

    @classmethod
    def from_proto(cls, proto: dlpack_pb2.DLPackVersion) -> DLPackVersion:
        return cls(
            major=proto.major,
            minor=proto.minor,
        )


class DLDeviceType(enum.Enum):
    """The enum that encodes the type of the device where
    DLTensor memory is allocated.
    """
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16


@dataclasses.dataclass
class DLDevice:
    """Represents the device where DLTensor memory is allocated.
    The device is represented by the pair of fields:
       device_type: DLDeviceType
       device_id: c_int
    """
    device_type: DLDeviceType
    device_id: int

    def to_proto(self) -> dlpack_pb2.DLDevice:
        return dlpack_pb2.DLDevice(
            device_type=self.device_type.value,
            device_id=self.device_id,
        )

    @classmethod
    def from_proto(cls, proto: dlpack_pb2.DLDevice) -> DLDevice:
        return cls(
            device_type=DLDeviceType(proto.device_type),
            device_id=proto.device_id,
        )


class DLDataTypeCode(enum.Enum):
    """An integer that encodes the category of DLTensor elements' data type."""
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaquePointer = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


@dataclasses.dataclass
class DLDataType:
    """Descriptor of data type for elements of DLTensor.
    The data type is described by a triple, `DLDataType.type_code`,
    `DLDataType.bits`, and `DLDataType.lanes`.

    The element is understood as packed `lanes` repetitions of
    elements from `type_code` data-category of width `bits`.
    """
    code: DLDataTypeCode
    bits: int
    lanes: int

    def to_proto(self) -> dlpack_pb2.DLDataType:
        return dlpack_pb2.DLDataType(
            code=self.code.value,
            bits=self.bits,
            lanes=self.lanes,
        )

    @classmethod
    def from_proto(cls, proto: dlpack_pb2.DLDataType) -> DLDataType:
        return cls(
            code=DLDataTypeCode(proto.code),
            bits=proto.bits,
            lanes=proto.lanes,
        )

    def to_numpy(self) -> npt.DTypeLike[Any]:
        import numpy as np
        type_map = {
            (DLDataTypeCode.kDLInt, 64, 1): np.int64,
            (DLDataTypeCode.kDLInt, 32, 1): np.int32,
            (DLDataTypeCode.kDLInt, 16, 1): np.int16,
            (DLDataTypeCode.kDLInt, 8, 1): np.int8,
            (DLDataTypeCode.kDLUInt, 64, 1): np.uint64,
            (DLDataTypeCode.kDLUInt, 32, 1): np.uint32,
            (DLDataTypeCode.kDLUInt, 16, 1): np.uint16,
            (DLDataTypeCode.kDLUInt, 8, 1): np.uint8,
            (DLDataTypeCode.kDLFloat, 64, 1): np.float64,
            (DLDataTypeCode.kDLFloat, 32, 1): np.float32,
            (DLDataTypeCode.kDLFloat, 16, 1): np.float16,
            (DLDataTypeCode.kDLComplex, 128, 1): np.complex128,
            (DLDataTypeCode.kDLComplex, 64, 1): np.complex64,
            (DLDataTypeCode.kDLBool, 8, 1): np.bool_,
        }
        return type_map[(self.code, self.bits, self.lanes)]

    def to_torch(self) -> torch.dtype:
        import torch
        type_map = {
            (DLDataTypeCode.kDLInt, 64, 1): torch.int64,
            (DLDataTypeCode.kDLInt, 32, 1): torch.int32,
            (DLDataTypeCode.kDLInt, 16, 1): torch.int16,
            (DLDataTypeCode.kDLInt, 8, 1): torch.int8,
            (DLDataTypeCode.kDLUInt, 8, 1): torch.uint8,
            (DLDataTypeCode.kDLFloat, 64, 1): torch.float64,
            (DLDataTypeCode.kDLFloat, 32, 1): torch.float32,
            (DLDataTypeCode.kDLFloat, 16, 1): torch.float16,
            (DLDataTypeCode.kDLComplex, 128, 1): torch.complex128,
            (DLDataTypeCode.kDLComplex, 64, 1): torch.complex64,
            (DLDataTypeCode.kDLBool, 8, 1): torch.bool,
        }
        return type_map[(self.code, self.bits, self.lanes)]

    def to_tensorflow(self) -> tf.DType:
        import tensorflow as tf
        type_map = {
            (DLDataTypeCode.kDLInt, 64, 1): tf.int64,
            (DLDataTypeCode.kDLInt, 32, 1): tf.int32,
            (DLDataTypeCode.kDLInt, 16, 1): tf.int16,
            (DLDataTypeCode.kDLInt, 8, 1): tf.int8,
            (DLDataTypeCode.kDLUInt, 64, 1): tf.uint64,
            (DLDataTypeCode.kDLUInt, 32, 1): tf.uint32,
            (DLDataTypeCode.kDLUInt, 16, 1): tf.uint16,
            (DLDataTypeCode.kDLUInt, 8, 1): tf.uint8,
            (DLDataTypeCode.kDLFloat, 64, 1): tf.float64,
            (DLDataTypeCode.kDLFloat, 32, 1): tf.float32,
            (DLDataTypeCode.kDLFloat, 16, 1): tf.float16,
            (DLDataTypeCode.kDLComplex, 128, 1): tf.complex128,
            (DLDataTypeCode.kDLComplex, 64, 1): tf.complex64,
            (DLDataTypeCode.kDLBool, 8, 1): tf.bool,
        }
        return type_map[(self.code, self.bits, self.lanes)]


@dataclasses.dataclass
class DLTensor:
    """Structure describing strided layout of DLTensor.
    Fields are:
       data:  void pointer
       device: DLDevice
       ndim: number of indices needed to reference an
             element of the tensor
       dtype: data type descriptor
       shape: tuple with lengths of the corresponding
              tensor dimensions
       strides: tuple of numbers of array elements to
                step in each dimension when traversing
                the tensor
       byte_offset: data + byte_offset gives the address of
                tensor element with index (0,) * ndim
    """
    data: bytes | bytearray | memoryview
    device: DLDevice
    ndim: int
    dtype: DLDataType
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    byte_offset: int

    @classmethod
    def from_dlpack(cls, arr: Any) -> DLTensor:
        """Convert an array to a DLTensorVersioned."""
        pycapsule = arr.__dlpack__()
        if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
            dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
                pycapsule, _c_str_dltensor,
            )
            dl_managed_tensor_ptr = ctypes.cast(
                dl_managed_tensor, ctypes.POINTER(C_DLManagedTensor),
            )
            dl_managed_tensor = dl_managed_tensor_ptr.contents

            dl_tensor = dl_managed_tensor.dl_tensor

            size = 1
            for idx in range(dl_tensor.ndim):
                size *= dl_tensor.shape[idx]
            size *= (dl_tensor.dtype.bits * dl_tensor.dtype.lanes + 7) // 8

            data = ctypes.pythonapi.PyMemoryView_FromMemory(
                dl_tensor.data, size, PyBUF_READ,
            )
            ndim = dl_tensor.ndim
            device = DLDevice(
                device_type=DLDeviceType(dl_tensor.device.device_type),
                device_id=dl_tensor.device.device_id,
            )
            dtype = DLDataType(
                code=DLDataTypeCode(dl_tensor.dtype.code),
                bits=dl_tensor.dtype.bits,
                lanes=dl_tensor.dtype.lanes,
            )
            shape = tuple(dl_tensor.shape[idx] for idx in range(ndim))
            strides = None
            if dl_tensor.strides:
                strides = tuple(dl_tensor.strides[idx] for idx in range(ndim))
            byte_offset = dl_tensor.byte_offset

            return cls(
                data=data,
                device=device,
                ndim=ndim,
                dtype=dtype,
                shape=shape,
                strides=strides,
                byte_offset=byte_offset,
            )
        raise ValueError("Expect a dltensor field, PyCapsule can only be consumed once.")

    def to_proto(self) -> dlpack_pb2.DLTensor:
        return dlpack_pb2.DLTensor(
            data=self.data.tobytes(),
            device=self.device.to_proto(),
            ndim=self.ndim,
            dtype=self.dtype.to_proto(),
            shape=self.shape,
            strides=self.strides,
            byte_offset=self.byte_offset,
        )

    @classmethod
    def from_proto(cls, proto: dlpack_pb2.DLTensor) -> DLTensor:
        return cls(
            data=proto.data,
            device=DLDevice.from_proto(proto.device),
            ndim=proto.ndim,
            dtype=DLDataType.from_proto(proto.dtype),
            shape=tuple(proto.shape),
            strides=tuple(proto.strides),
            byte_offset=proto.byte_offset,
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        import numpy as np
        return np.frombuffer(
            self.data, dtype=self.dtype.to_numpy(),
        ).reshape(self.shape)

    def to_torch(self) -> Any:
        import torch
        torch.frombuffer(
            self.data, dtype=self.dtype.to_torch(),
        ).reshape(self.shape)

    def to_tensorflow(self) -> Any:
        import tensorflow as tf
        return tf.reshape(
            tf.io.decode_raw(self.data, self.dtype.to_tensorflow()),
            self.shape,
        )
