from graph_kmer_index.shared_mem import to_shared_memory, from_shared_memory
from graph_kmer_index import KmerIndex

index = KmerIndex.from_file("testdata2_index.npz")
print(index.get(852840309094508953))

to_shared_memory(index, "testindex")

new_index = from_shared_memory(KmerIndex, "testindex")
print(new_index.get(852840309094508953))
