use std::sync::Arc;

use dlpark::prelude::{ManagedTensor, ManagerCtx};
use dlpark::TensorView;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub mod convert;
pub mod gen;
mod tensor_container;

use crate::convert::tensor_to_fb;
use crate::gen::Tensor::dlpack::Tensor;
use crate::tensor_container::TensorContainer;

#[pyfunction]
fn dlpack_to_bytes(py: Python<'_>, tensor: ManagedTensor) -> PyResult<&PyBytes> {
    let fbb = tensor_to_fb(&tensor);
    let metadata = fbb.finished_data();
    let mut data = (metadata.len() as u32).to_le_bytes().to_vec();
    data.extend_from_slice(fbb.finished_data());
    data.extend_from_slice(unsafe {
        std::slice::from_raw_parts(tensor.data_ptr() as *mut u8, tensor.data_size())
    });
    let py_bytes = PyBytes::new_with(py, data.len(), |b: &mut [u8]| Ok(b.copy_from_slice(&data)));
    py_bytes
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
