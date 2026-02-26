using Squiggles
using CUDA

Squiggles.set_GPUbackend()

data = rand(256, 100)

cc(data, data, 256, 256)
