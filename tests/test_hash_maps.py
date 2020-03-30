from graph_kmer_index.logn_hash_map import ModuloHashMap


def test_modulo_hashmap():
    values = [1, 5, 6, 10, 40, 45, 452930477 + 100]
    hashmap = ModuloHashMap.from_sorted_array(values)

    assert hashmap.hash(10) == 3
    assert hashmap.hash(5) == 1
    assert hashmap.hash(452930477 + 100) == 6


if __name__ == "__main__":
    test_modulo_hashmap()