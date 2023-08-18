import ctypes

_c_str_dltensor = b"dltensor"


class DLPackVersion(ctypes.Structure):
    """Structure representing the version of DLPack."""
    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
    ]


class DLDevice(ctypes.Structure):
    """Represents the device where DLTensor memory is allocated.
    The device is represented by the pair of fields:
       device_type: DLDeviceType
       device_id: c_int
    """
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    """Descriptor of data type for elements of DLTensor.
    The data type is described by a triple, `DLDataType.type_code`,
    `DLDataType.bits`, and `DLDataType.lanes`.

    The element is understood as packed `lanes` repetitions of
    elements from `type_code` data-category of width `bits`.
    """
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
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
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    """Structure storing the pointer to the tensor descriptor,
    deleter callable for the tensor descriptor, and pointer to
    some additional data. These are stored in fields `dl_tensor`,
    `deleter`, and `manager_ctx`."""
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]
