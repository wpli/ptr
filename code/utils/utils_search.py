import collections
import cPickle
import sys
import heapq
def get_ranked_matches( query_word_set, word_inverted_index, top_k=10 ):
    all_indices = []
    for word in list( query_word_set ):
        indices = word_inverted_index[word]
        all_indices += indices

    count_by_index = collections.Counter( all_indices )
    
    k_keys_sorted = heapq.nlargest(top_k, count_by_index, key=count_by_index.get)
    return [ (i,count_by_index[i]) for i in k_keys_sorted]

def unit_test():
    with open( '../../data/processed/word_inverted_index.pkl' ) as f:
        word_inverted_index = cPickle.load( f )
    
    query_word_set = { 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog' }
    get_ranked_matches( query_word_set, word_inverted_index )

if __name__ == '__main__':
    unit_test()
