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
    cdef long modulo
    cdef unsigned long *hashes
    cdef unsigned int[:] node_hits
    cdef unsigned short[:] frequencies
    cdef unsigned long[:] ref_offsets_hits
    cdef unsigned int[:] read_offsets_hits

    def __init__(self, index):
        self.hashes_to_index = index._hashes_to_index
        self.n_kmers = index._n_kmers
        self.nodes = index._nodes
        self.ref_offsets = index._ref_offsets
        self.kmers = index._kmers
        self.frequencies = index._frequencies
        self.modulo = index._modulo
        self.ref_offsets_hits = np.empty(0, dtype=np.uint64)
        self.node_hits = np.empty(0, dtype=np.uint32)
        self.read_offsets_hits = np.empty(0, dtype=np.uint32)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.unraisable_tracebacks(True)
    cpdef int get(self, unsigned long[:] kmers):
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

                if self.frequencies[index_position+j] > 5:
                    continue
                n_total_hits += 1

        self.node_hits = np.empty(n_total_hits, dtype=np.uint32)
        self.ref_offsets_hits = np.empty(n_total_hits, dtype=np.uint64)
        self.read_offsets_hits = np.empty(n_total_hits, dtype=np.uint32)

        if n_total_hits == 0:
            return 0

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
                if self.frequencies[index_position+j] > 5:
                    continue
                self.node_hits[counter] = self.nodes[index_position+j]
                self.ref_offsets_hits[counter] = self.ref_offsets[index_position+j]
                self.read_offsets_hits[counter] = i
                counter += 1

    cpdef unsigned long[:] get_ref_offsets_hits(self):
        return self.ref_offsets_hits

    cpdef unsigned int[:] get_node_hits(self):
        return self.node_hits

    cpdef unsigned int[:] get_read_offsets_hits(self):
        return self.read_offsets_hits


cpdef test(index):
    kmers = np.arange(200000000, 200000000 + 119, dtype=np.uint64)
    start_time = time.time()
    index.get(kmers)

    end_time = time.time()
    print("Time spent: ", end_time - start_time)


