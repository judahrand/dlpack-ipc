import copy
import time

import torch
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal

import dlpack


arr = np.random.random((3, 4)).astype(dtype=np.float32)

dlpack.serialize_dlpack(np.array([]).__dlpack__())  # Force numpy to initialize
ts = time.perf_counter()
buf = dlpack.serialize_dlpack(arr.__dlpack__())
duration = (time.perf_counter() - ts) * 1e6
print(f"Serialization took: {duration:.3f} us")

np.from_dlpack(np.array([]))
ts = time.perf_counter()
numpy_array = dlpack.bytes_to_numpy(buf)
duration = (time.perf_counter() - ts) * 1e6
assert_array_equal(arr, numpy_array)
print(f"Deserialization to Numpy took: {duration:.3f} us")

torch.from_dlpack(np.array([]))  # Force torch to initialize
ts = time.perf_counter()
torch_tensor = dlpack.bytes_to_torch(buf)
duration = (time.perf_counter() - ts) * 1e6
assert_array_equal(arr, torch_tensor.numpy())
print(f"Deserialization to Torch took: {duration:.3f} us")

tf.experimental.dlpack.from_dlpack(np.array([]).__dlpack__())  # Force tensorflow to initialize
ts = time.perf_counter()
tensorflow_tensor = dlpack.bytes_to_tensorflow(buf)
duration = (time.perf_counter() - ts) * 1e6
assert_array_equal(arr, tensorflow_tensor.numpy())
print(f"Deserialization to Tensorflow took: {duration:.3f} us")
