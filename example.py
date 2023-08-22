import time

import dlpack
import numpy as np


arr = np.random.random((3, 4)).astype(dtype=np.float32)

ts = time.perf_counter()
buf = dlpack.serialize_dlpack(arr.__dlpack__())
duration = (time.perf_counter() - ts) * 1e6
print(f"Serialization took: {duration:.3f} us")

ts = time.perf_counter()
numpy_array = dlpack.bytes_to_numpy(buf)
duration = (time.perf_counter() - ts) * 1e6
print(f"Deserialization to Numpy took: {duration:.3f} us")
print(numpy_array)

ts = time.perf_counter()
torch_tensor = dlpack.bytes_to_torch(buf)
duration = (time.perf_counter() - ts) * 1e6
print(f"Deserialization to Torch took: {duration:.3f} us")
print(torch_tensor)

ts = time.perf_counter()
tensorflow_tensor = dlpack.bytes_to_tensorflow(buf)
duration = (time.perf_counter() - ts) * 1e6
print(f"Deserialization to Tensorflow took: {duration:.3f} us")
print(tensorflow_tensor)
