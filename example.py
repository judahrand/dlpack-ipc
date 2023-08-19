import dlpack
import numpy as np

arr = np.random.random((10000, 3, 4))

dl_tensor = dlpack.DLTensor.from_dlpack(arr)

# Serialize to protobuf
proto = dl_tensor.to_proto()

# Deserialize from protobuf
dl_tensor = dlpack.DLTensor.from_proto(proto)

# Convert to numpy
dl_tensor.to_numpy()

# Convert to pytorch
dl_tensor.to_torch()

# Convert to tensorflow
dl_tensor.to_tensorflow()
