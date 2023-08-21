use dlpark::prelude::{ManagedTensor, TensorView};
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::gen;

pub fn dtype_to_fb_offset<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    dtype: &dlpark::ffi::DataType,
) -> WIPOffset<gen::Tensor::dlpack::DataType<'a>> {
    let mut builder = gen::Tensor::dlpack::DataTypeBuilder::new(fbb);
    builder.add_code(dtype.code as u8);
    builder.add_bits(dtype.bits as u8);
    builder.add_lanes(dtype.lanes as u16);
    builder.finish()
}

pub fn tensor_to_fb_offset<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    tensor: &dlpark::ManagedTensor,
) -> WIPOffset<gen::Tensor::dlpack::Tensor<'a>> {
    let fb_shape = fbb.create_vector(tensor.shape());
    let fb_strides = match tensor.strides() {
        Some(strides) => Some(fbb.create_vector(strides)),
        None => None,
    };
    let fb_dtype = dtype_to_fb_offset(fbb, &tensor.dtype());

    let mut builder = gen::Tensor::dlpack::TensorBuilder::new(fbb);
    builder.add_ndim(tensor.ndim() as i32);
    builder.add_dtype(fb_dtype);
    builder.add_shape(fb_shape);
    match fb_strides {
        Some(_) => {
            builder.add_strides(fb_strides.unwrap());
        }
        None => {}
    }
    builder.add_byte_offset(tensor.byte_offset());
    builder.finish()
}

pub fn tensor_to_fb(tensor: &ManagedTensor) -> FlatBufferBuilder {
    let mut fbb = FlatBufferBuilder::new();

    let root = tensor_to_fb_offset(&mut fbb, tensor);

    fbb.finish(root, None);

    fbb
}
