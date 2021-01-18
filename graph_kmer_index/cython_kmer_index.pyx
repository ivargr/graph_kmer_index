import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from cython.parallel import prange
import time
from libc.stdlib cimport malloc, free


@cython.cdivision(True)
@cython.boundscheck(False)
cdef unsigned long[:] modulo_of_array(unsigned long[:] a, int modulo):
    cdef int i
    cdef int size = a.shape[0]
    cdef unsigned long[:] m = np.empty(size, dtype=np.uint64)
    for i in range(a.shape[0]):
        m[i] = a[i] % modulo
    return m


@cython.final
cdef class CythonKmerIndex:
    cdef long[:] hashes_to_index
    cdef unsigned int[:] n_kmers
    cdef unsigned int[:] nodes
    cdef unsigned long[:] ref_offsets
    cdef unsigned long[:] kmers
    cdef float[:] allele_frequencies
    cdef long modulo
    cdef unsigned long *hashes
    cdef unsigned short[:] frequencies

    def __init__(self, index):
        print("Init cython kmer index")
        self.hashes_to_index = index._hashes_to_index
        self.n_kmers = index._n_kmers
        self.nodes = index._nodes
        self.ref_offsets = index._ref_offsets
        self.kmers = index._kmers
        self.frequencies = index._frequencies
        self.allele_frequencies = index._allele_frequencies
        self.modulo = index._modulo

    @cython.cdivision(True)
    #@cython.boundscheck(False)
    @cython.unraisable_tracebacks(True)
    cpdef np.ndarray[np.uint64_t, ndim=2] get(self, unsigned long[:] kmers):
        cdef int n = kmers.shape[0]
        cdef unsigned long[:] kmer_hashes = modulo_of_array(kmers, self.modulo)
        cdef int n_total_hits = 0
        cdef int i, j
        cdef unsigned long hash
        cdef unsigned int n_local_hits
        cdef long index_position

        # First find number of hits
        for i in range(n):
            hash = kmer_hashes[i]
            if hash == 0:
                continue
            n_local_hits = self.n_kmers[hash]
            if n_local_hits > 10000:
                continue

            index_position = self.hashes_to_index[hash]
            for j in range(n_local_hits):
                if self.kmers[index_position+j] != kmers[i]:
                    continue

                if self.frequencies[index_position+j] > 20:
                    continue
                n_total_hits += 1

        cdef np.ndarray[np.uint64_t, ndim=2] output_data = np.zeros((5, n_total_hits), dtype=np.uint64)

        if n_total_hits == 0:
            output_data

        # Get the actual hits
        cdef int counter = 0

        for i in range(n):
            hash = kmer_hashes[i]
            if hash == 0:
                continue

            index_position = self.hashes_to_index[hash]
            n_local_hits = self.n_kmers[hash]

            if n_local_hits == 0:
                continue

            if n_local_hits > 10000:
                continue

            for j in range(n_local_hits):
                if self.kmers[index_position+j] != kmers[i]:
                    continue
                if self.frequencies[index_position+j] > 20:
                    continue
                output_data[0, counter] = self.nodes[index_position+j]
                output_data[1, counter] = self.ref_offsets[index_position+j]
                output_data[2, counter] = i
                output_data[3, counter] = self.frequencies[index_position+j]
                output_data[4, counter] = (<np.uint64_t> (1000 * self.allele_frequencies[index_position+j]))

                counter += 1

        return output_data


cpdef test(index):
    kmers = np.arange(200000000, 200000000 + 119, dtype=np.uint64)
    start_time = time.time()
    index.get(kmers)

    end_time = time.time()
    print("Time spent: ", end_time - start_time)


