import dlpack
import numpy as np

arr = np.random.random((3, 4)).astype(dtype=np.float32)

buf = dlpack.serialize_dlpack(arr.__dlpack__())

numpy_array = dlpack.bytes_to_numpy(buf)
print(numpy_array)

torch_tensor = dlpack.bytes_to_torch(buf)
print(torch_tensor)

tensorflow_tensor = dlpack.bytes_to_tensorflow(buf)
print(tensorflow_tensor)
