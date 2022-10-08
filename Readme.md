

# Graph Kmer Index
This is a collection of scripts and modules that can be used to sample kmers from graphs. Graph Kmer Index is not directly meant to be used alone, but is used as part of the [KAGE](https://github.com/ivargr/kage) genotyper.



### Finding all kmers in a graph
The class `DenseKmerFinder` can be used to find all kmers in an `obgraph`:


#### Step 1: Create a graph

The easiest method for testing is just creating a graph in Python like this:
```python
from obgraph import Graph

# We first define a graph
graph = Graph.from_dicts(
    node_sequences={
        1: "ACTG",
        2: "A",
        3: "G",
        4: "CCCC"
     },
    edges={
        1: [2, 3],
        2: [4],
        4: [4]
    },
    linear_ref_nodes=[1, 2, 4]  # required, denotes the linear ref path through the graph
)
```


Alternatively, a graph can be created from a GFA like this:
```bash
# Convert GFA node ids to numeric IDs (necessary for obgraph)
obgraph convert_gfa_ids_to_numeric -g graph.gfa -o numeric.gfa
# Create obgraph
obgraph from_gfa -g numeric.gfa -o graph
```

The above will create a `graph.npz` file which can be read in Python:Ø
```python
from obgraph import Graph
graph = Graph.from_file("graph.npz")
```



#### Step 2: Get kmers
```python
from graph_kmer_index.kmer_finder import DenseKmerFinder
finder = DenseKmerFinder(graph, k=5)
finder.find()
kmers, nodes = finder.get_found_kmers_and_nodes()

# kmers and nodes are now one kmer and one node for every combination of kmer/node
# e.g. if one kmer touches two nodes, it will be listed twice with the two nodes

print(kmers)
print(nodes)

```

This should give:
```
[ 97  97  97 389 389 389 109 109 109 437 437 437]
[1 2 4 1 2 4 1 3 4 1 3 4]
```

... or in a more readable format:
```python
from graph_kmer_index import kmer_hash_to_sequence
for kmer, node in zip(kmers, nodes):
    print(kmer_hash_to_sequence(kmer, 5), node)
```

```python
actac 1
actac 2
actac 4
ctacc 1
ctacc 2
ctacc 4
actgc 1
actgc 3
actgc 4
ctgcc 1
ctgcc 3
ctgcc 4
```