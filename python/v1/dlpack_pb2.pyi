from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DLPackVersion(_message.Message):
    __slots__ = ["major", "minor"]
    MAJOR_FIELD_NUMBER: _ClassVar[int]
    MINOR_FIELD_NUMBER: _ClassVar[int]
    major: int
    minor: int
    def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ...) -> None: ...

class DLDevice(_message.Message):
    __slots__ = ["device_type", "device_id"]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_type: int
    device_id: int
    def __init__(self, device_type: _Optional[int] = ..., device_id: _Optional[int] = ...) -> None: ...

class DLDataType(_message.Message):
    __slots__ = ["code", "bits", "lanes"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    BITS_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    code: int
    bits: int
    lanes: int
    def __init__(self, code: _Optional[int] = ..., bits: _Optional[int] = ..., lanes: _Optional[int] = ...) -> None: ...

class DLTensor(_message.Message):
    __slots__ = ["data", "device", "ndim", "dtype", "shape", "strides", "byte_offset"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    NDIM_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    STRIDES_FIELD_NUMBER: _ClassVar[int]
    BYTE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    device: DLDevice
    ndim: int
    dtype: DLDataType
    shape: _containers.RepeatedScalarFieldContainer[int]
    strides: _containers.RepeatedScalarFieldContainer[int]
    byte_offset: int
    def __init__(self, data: _Optional[bytes] = ..., device: _Optional[_Union[DLDevice, _Mapping]] = ..., ndim: _Optional[int] = ..., dtype: _Optional[_Union[DLDataType, _Mapping]] = ..., shape: _Optional[_Iterable[int]] = ..., strides: _Optional[_Iterable[int]] = ..., byte_offset: _Optional[int] = ...) -> None: ...

class DLTensorVersioned(_message.Message):
    __slots__ = ["version", "flags", "dl_tensor"]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    DL_TENSOR_FIELD_NUMBER: _ClassVar[int]
    version: DLPackVersion
    flags: int
    dl_tensor: DLTensor
    def __init__(self, version: _Optional[_Union[DLPackVersion, _Mapping]] = ..., flags: _Optional[int] = ..., dl_tensor: _Optional[_Union[DLTensor, _Mapping]] = ...) -> None: ...
