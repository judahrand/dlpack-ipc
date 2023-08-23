use dlpark::prelude::{ManagedTensor, ManagerCtx};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::buffer::{Buffer, MutableBuffer};

use crate::convert::{tensor_to_bytes, write_message};
use crate::gen::Tensor::dlpack::Tensor;
use crate::tensor_container::TensorContainer;

#[pyclass]
struct PyBuffer {
    buffer: Buffer,
}

impl From<Buffer> for PyBuffer {
    fn from(buffer: Buffer) -> Self {
        PyBuffer { buffer }
    }
}

#[pymethods]
impl PyBuffer {
    unsafe fn __getbuffer__(
        slf: &PyCell<Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        fill_view_from_readonly_data(view, flags, &slf.borrow().buffer.as_slice(), slf)
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // Release memory held by the format string
        drop(std::ffi::CString::from_raw((*view).format));
    }

    fn to_pybytes<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new_with(
            py,
            self.buffer.len(),
            |b: &mut [u8]| {
                b.copy_from_slice(self.buffer.as_slice());
                Ok(())
            },
        )?)
    }
}

#[pyfunction]
fn py_buffer(py: Python<'_>, obj: &PyAny) -> PyResult<PyBuffer> {
    let buf = pyo3::buffer::PyBuffer::<u8>::get(obj)?.to_vec(py)?;
    Ok(Buffer::from(buf).into())
}

#[pyfunction]
fn write_tensor_to_buffer(tensor: ManagedTensor) -> PyResult<PyBuffer> {
    let encoded = tensor_to_bytes(&tensor);
    let buffer = write_message(&encoded);
    Ok(buffer.into())
}

#[pyfunction]
fn read_tensor_from_buffer(py_buffer: &PyBuffer) -> PyResult<ManagerCtx<TensorContainer>> {
    let data = py_buffer.buffer.as_slice();
    fn pop(b: &[u8]) -> [u8; 4] {
        b.try_into().expect("slice with incorrect length")
    }
    let metadata_len = u32::from_le_bytes(pop(&data[..4])) as usize;
    let metadata = &data[4..metadata_len + 4];
    let tensor = flatbuffers::root::<Tensor>(metadata).unwrap();
    let tensor_container = TensorContainer::new(
        Buffer::from(&data[metadata_len + 4..]),
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
    m.add_class::<PyBuffer>()?;
    m.add_class::<MutableBuffer>()?;
    Ok(())
}

/// # Safety
///
/// `view` must be a valid pointer to ffi::Py_buffer, or null
/// `data` must outlive the Python lifetime of `owner` (i.e. data must be owned by owner, or data
/// must be static data)
unsafe fn fill_view_from_readonly_data(
    view: *mut pyo3::ffi::Py_buffer,
    flags: std::ffi::c_int,
    data: &[u8],
    owner: &PyAny,
) -> PyResult<()> {
    if view.is_null() {
        return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
    }

    if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
        return Err(pyo3::exceptions::PyBufferError::new_err(
            "Object is not writable",
        ));
    }

    (*view).obj = pyo3::ffi::_Py_NewRef(owner.into_ptr());

    (*view).buf = data.as_ptr() as *mut std::ffi::c_void;
    (*view).len = data.len() as isize;
    (*view).readonly = 1;
    (*view).itemsize = 1;

    (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
        let msg = std::ffi::CString::new("B").unwrap();
        msg.into_raw()
    } else {
        std::ptr::null_mut()
    };

    (*view).ndim = 1;
    (*view).shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
        &mut (*view).len
    } else {
        std::ptr::null_mut()
    };

    (*view).strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
        &mut (*view).itemsize
    } else {
        std::ptr::null_mut()
    };

    (*view).suboffsets = std::ptr::null_mut();
    (*view).internal = std::ptr::null_mut();

    Ok(())
}
