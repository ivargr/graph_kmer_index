import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from cython.parallel import prange
import time
from libc.stdlib cimport malloc, free


cdef class CythonReferenceKmerIndex:
    cdef unsigned long[:] ref_position_to_index
    cdef unsigned long[:] kmers


    def __init__(self, index):
        self.ref_position_to_index = index.ref_position_to_index
        self.kmers = index.kmers

    @cython.boundscheck(False)
    cpdef np.ndarray[np.uint64_t, ndim=1] get_between(self, unsigned long ref_start, unsigned long ref_end):
        cdef unsigned long index_start = self.ref_position_to_index[ref_start]
        cdef unsigned long index_end = self.ref_position_to_index[ref_end]

        cdef np.ndarray[np.uint64_t, ndim=1] output_data = np.zeros(index_end - index_start, dtype=np.uint64)
        cdef unsigned long i
        for i in range(index_end - index_start):
            output_data[i] = self.kmers[index_start + i]


        return output_data
