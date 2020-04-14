import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple get_nodes_and_ref_offsets_from_multiple_kmers(np.ndarray[np.int64_t] kmers,
                                                  np.ndarray[np.int64_t] hashes,
                                                  np.ndarray[np.int64_t] hashes_to_index,
                                                  np.ndarray[np.int64_t] n_kmers,
                                                  np.ndarray[np.uint32_t] nodes,
                                                  np.ndarray[np.uint32_t] ref_offsets):

    cdef np.ndarray[np.int64_t, ndim=1] kmer_hashes = np.zeros(kmers.shape[0], dtype=np.int)
    kmer_hashes = np.mod(kmers, 452930477)

    cdef int index_position, i

    # First find number of hits
    cdef int n_total_hits = 0
    cdef int hash
    cdef int n = kmers.shape[0]
    for i in range(n):
        hash = hashes[kmer_hashes[i]]
        if hash == 0:
            continue
        n_total_hits += n_kmers[hash]


    # Get the actual hits
    cdef np.ndarray[np.int64_t] found_nodes = np.zeros(n_total_hits, dtype=np.int)
    cdef np.ndarray[np.int64_t] found_ref_offsets = np.zeros(n_total_hits, dtype=np.int)
    cdef np.ndarray[np.int64_t] found_read_offsets = np.zeros(n_total_hits, dtype=np.int)

    cdef int counter = 0
    cdef int n_local_hits, j
    for i in range(n):
        hash = hashes[kmer_hashes[i]]
        if hash == 0:
            continue

        index_position = hashes_to_index[hash]
        n_local_hits = n_kmers[hash]

        if n_local_hits == 0:
            continue

        for j in range(n_local_hits):
            found_nodes[counter] = nodes[index_position+j]
            found_ref_offsets[counter] = ref_offsets[index_position+j]
            found_read_offsets[counter] = i
            counter += 1


    return found_nodes, found_ref_offsets, found_read_offsets
