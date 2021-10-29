from graph_kmer_index import kmer_hash_to_sequence, sequence_to_kmer_hash


def test():
    sequence = "atg"
    hash = sequence_to_kmer_hash(sequence)
    print(hash)
    sequence2 = kmer_hash_to_sequence(hash, len(sequence))
    print(sequence2)

    sequence = "ggtagctctcgccagctcctagaaggagga"
    hash = sequence_to_kmer_hash(sequence)
    print(hash)
    sequence2 = kmer_hash_to_sequence(hash, len(sequence))
    print(sequence2)


    # todo:
    #assert sequence2 == sequence, "%s != %s" % (sequence2, sequence)

test()