use std::sync::Arc;

use dlpark::prelude::{ManagedTensor, ManagerCtx};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub mod convert;
pub mod gen;
mod tensor_container;

use crate::convert::{tensor_to_bytes, write_message};
use crate::gen::Tensor::dlpack::Tensor;
use crate::tensor_container::TensorContainer;

#[pyfunction]
fn dlpack_to_bytes(py: Python<'_>, tensor: ManagedTensor) -> PyResult<&PyBytes> {
    let encoded = tensor_to_bytes(&tensor);
    let buf = write_message(&encoded);
    Ok(unsafe{PyBytes::from_ptr(py, buf.as_ptr(), buf.len())})
}

#[pyfunction]
fn bytes_to_dlpack<'a>(bytes: &PyBytes) -> PyResult<ManagerCtx<TensorContainer>> {
    let data = bytes.as_bytes();
    fn pop(barry: &[u8]) -> [u8; 4] {
        barry.try_into().expect("slice with incorrect length")
    }
    let metadata_len = u32::from_le_bytes(pop(&data[..4])) as usize;
    let metadata = &data[4..metadata_len + 4];
    let tensor = flatbuffers::root::<Tensor>(metadata).unwrap();
    let tensor_container = TensorContainer::new(
        Arc::new(data[metadata_len + 4..].to_vec()),
        tensor.dtype().into(),
        tensor.shape().into_iter().map(|x| x).collect::<Vec<i64>>(),
        match tensor.strides() {
            Some(strides) => Some(strides.into_iter().map(|x| x).collect::<Vec<i64>>()),
            None => None,
        },
        tensor.byte_offset(),
    );
    Ok(ManagerCtx::new(tensor_container))
}

#[pymodule]
fn dlpack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dlpack_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(bytes_to_dlpack, m)?)?;
    Ok(())
}
