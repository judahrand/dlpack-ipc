use crate::gen::Tensor::dlpack::DataType as DataTypeFb;
use dlpark::ffi::{DataType, Device, DeviceType};
use dlpark::prelude::{CowIntArray, ToTensor};

pub struct DataTypeContainer {
    code: u8,
    bits: u8,
    lanes: u16,
}

impl From<DataTypeFb<'_>> for DataTypeContainer {
    fn from(dtype: DataTypeFb) -> Self {
        Self {
            code: dtype.code() as u8,
            bits: dtype.bits(),
            lanes: dtype.lanes(),
        }
    }
}

pub struct TensorContainer {
    data: *const u8,
    dtype: DataTypeContainer,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    byte_offset: u64,
}

impl TensorContainer {
    pub fn new(
        data: *const u8,
        dtype: DataTypeContainer,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
        byte_offset: u64,
    ) -> Self {
        Self {
            data,
            dtype,
            shape,
            strides,
            byte_offset,
        }
    }
}

impl ToTensor for TensorContainer {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.data as *mut std::ffi::c_void
    }

    fn shape(&self) -> CowIntArray {
        CowIntArray::from_owned(self.shape.clone())
    }

    fn strides(&self) -> Option<CowIntArray> {
        if self.strides.is_none() {
            None
        } else {
            Some(CowIntArray::from_owned(self.strides.clone().unwrap()))
        }
    }

    fn device(&self) -> Device {
        Device {
            device_type: DeviceType::Cpu,
            device_id: 0,
        }
    }

    fn dtype(&self) -> DataType {
        DataType {
            code: unsafe { std::mem::transmute(self.dtype.code) },
            bits: self.dtype.bits,
            lanes: self.dtype.lanes,
        }
    }

    fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
}
