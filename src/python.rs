use dlpark::prelude::{ManagedTensor, ManagerCtx};
use pyo3::prelude::*;

use crate::buffer::{Buffer, MutableBuffer};

use crate::convert::{tensor_to_bytes, write_message};
use crate::gen::Tensor::dlpack::Tensor;
use crate::tensor_container::TensorContainer;

#[pyfunction]
fn py_buffer(_py: Python, obj: &PyAny) -> PyResult<MutableBuffer> {
    let buf = pyo3::buffer::PyBuffer::<u8>::get(obj).unwrap();
    let mut res = MutableBuffer::new(0);
    res.extend_from_slice(&buf.to_vec(_py).unwrap()[..]);
    Ok(res)
}

#[pyfunction]
fn write_tensor_to_buffer(tensor: ManagedTensor, buffer: &mut MutableBuffer) -> PyResult<()> {
    let encoded = tensor_to_bytes(&tensor);
    write_message(&encoded, buffer);
    Ok(())
}

#[pyfunction]
fn read_tensor_from_buffer<'a>(buffer: &MutableBuffer) -> PyResult<ManagerCtx<TensorContainer>> {
    let data = buffer.as_slice();
    fn pop(barry: &[u8]) -> [u8; 4] {
        barry.try_into().expect("slice with incorrect length")
    }
    let metadata_len = u32::from_le_bytes(pop(&data[..4])) as usize;
    let metadata = &data[4..metadata_len + 4];
    let tensor = flatbuffers::root::<Tensor>(metadata).unwrap();
    let tensor_container = TensorContainer::new(
        data[metadata_len + 4..].as_ptr(),
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
    m.add_function(wrap_pyfunction!(py_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(write_tensor_to_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(read_tensor_from_buffer, m)?)?;
    m.add_class::<Buffer>()?;
    m.add_class::<MutableBuffer>()?;
    Ok(())
}
