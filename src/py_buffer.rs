// use pyo3::ffi;
// use pyo3::prelude::*;
// use pyo3::AsPyPointer;
// use crate::bytes::Bytes;
// use std::ffi::{c_int, c_void, CString};
// use pyo3::exceptions::PyBufferError;
// use std::ptr;
// use std::ptr::NonNull;

// use crate::bytes::Bytes;

// #[pyclass]
// pub struct Buffer {
//     data: Arc<Bytes>,
//     ptr: *const u8,
//     length: usize,
// }

// impl Buffer {
//     pub fn new(data: Arc<Bytes>, ptr: *const u8, length: usize) -> Self {
//         Self {
//             data,
//             ptr,
//             length,
//         }
//     }
// }

// #[pymethods]
// impl Buffer {
//     unsafe fn __getbuffer__(
//         slf: &PyCell<Self>,
//         view: *mut ffi::Py_buffer,
//         flags: c_int,
//     ) -> PyResult<()> {
//         fill_view_from_readonly_data(
//             view, flags, &slf.borrow().data.as_ref(), slf,
//         )
//     }

//     unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
//         // Release memory held by the format string
//         drop(CString::from_raw((*view).format));
//     }
// }

// /// # Safety
// ///
// /// `view` must be a valid pointer to ffi::Py_buffer, or null
// /// `data` must outlive the Python lifetime of `owner` (i.e. data must be owned by owner, or data
// /// must be static data)
// unsafe fn fill_view_from_readonly_data(
//     view: *mut ffi::Py_buffer,
//     flags: c_int,
//     data: &[u8],
//     owner: &PyAny,
// ) -> PyResult<()> {
//     if view.is_null() {
//         return Err(PyBufferError::new_err("View is null"));
//     }

//     if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
//         return Err(PyBufferError::new_err("Object is not writable"));
//     }

//     (*view).obj = ffi::_Py_NewRef(owner.as_ptr());

//     (*view).buf = data.as_ptr() as *mut c_void;
//     (*view).len = data.len() as isize;
//     (*view).readonly = 1;
//     (*view).itemsize = 1;

//     (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
//         let msg = CString::new("B").unwrap();
//         msg.into_raw()
//     } else {
//         ptr::null_mut()
//     };

//     (*view).ndim = 1;
//     (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
//         &mut (*view).len
//     } else {
//         ptr::null_mut()
//     };

//     (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
//         &mut (*view).itemsize
//     } else {
//         ptr::null_mut()
//     };

//     (*view).suboffsets = ptr::null_mut();
//     (*view).internal = ptr::null_mut();

//     Ok(())
// }
