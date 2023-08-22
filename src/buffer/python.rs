use pyo3::prelude::*;

use super::Buffer;
use super::MutableBuffer;

#[pymethods]
impl Buffer {
    unsafe fn __getbuffer__(
        slf: &PyCell<Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        fill_view_from_readonly_data(view, flags, &slf.borrow().as_slice(), slf)
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // Release memory held by the format string
        drop(std::ffi::CString::from_raw((*view).format));
    }
}

#[pymethods]
impl MutableBuffer {
    unsafe fn __getbuffer__(
        slf: &PyCell<Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        fill_view_from_readonly_data(view, flags, &slf.borrow().as_slice(), slf)
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // Release memory held by the format string
        drop(std::ffi::CString::from_raw((*view).format));
    }
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
