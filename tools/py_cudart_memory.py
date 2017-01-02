#!/usr/bin/env python

import ctypes

# Path to location of libcudart
_CUDA = "/usr/local/cuda-7.5/lib64/libcudart.so"
cuda = ctypes.cdll.LoadLibrary(_CUDA)

cuda.cudaMemGetInfo.restype = int
cuda.cudaMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

cuda.cudaGetErrorString.restype = ctypes.c_char_p
cuda.cudaGetErrorString.argtypes = [ctypes.c_int]


def cudaMemGetInfo(mb=False):
    """
    Return (free, total) memory stats for CUDA GPU
    Default units are bytes. If mb==True, return units in MB
    """
    print 'gpu: '
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    ret = cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))

    if ret != 0:
        err = cuda.cudaGetErrorString(status)
        raise RuntimeError("CUDA Error (%d): %s" % (status, err))

    if mb:
        scale = 1024.0**2
        return free.value / scale, total.value / scale
    else:
        return free.value, total.value

free_value, total_value = cudaMemGetInfo(mb=True)
print total_value - free_value, '/', total_value
