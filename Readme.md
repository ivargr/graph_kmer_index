



## Make a flat index
This only finds kmers and associated nodes and stores in two numpy arrays
```python
graph_kmer_index make ...
```

## Merge multiple flat indexes into a index that can be used to do lookup
```python
graph_kmer_index merge_indexes -o out_file_name index1.npz index2.npz ...
```


