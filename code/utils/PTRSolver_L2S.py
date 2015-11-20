# FUNCTIONS
import collections
import sys
import parameter_operations
def get_n_ngram_counts( wordids, exclusion_set = set() ):
    n_ngram_counts = {}
    for n in range(1,50):
        sys.stderr.write( "%s " % n )
        ngrams = []
        for doc in wordids:
            for i in range( len( doc ) - n + 1 ):
                tup = tuple( doc[i:i+n] )
                ngrams.append( tup )

        n_ngram_counts[n] = collections.Counter( ngrams )

    return n_ngram_counts

def get_ngrams_to_exclude( ngram_size, idea ):
    ngrams_to_exclude = []
    for i in range( len( idea ) - ngram_size + 1 ):
        ngrams_to_exclude.append( tuple( idea[i:i+ngram_size] ) )
            
    return ngrams_to_exclude
    

def get_exclusion_set( ngram_size, idea ):
    exclusion_set = set()

    for i in range( len( idea ) - ngram_size + 1 ):
        exclusion_set.add( tuple( idea[i:i+ngram_size] ) )
            
    return exclusion_set

def get_ngram_maxcount( n_ngram_counts ):
    ngram_maxcount = []
    for n in range( 3, 50 ):
        maxcount = max( n_ngram_counts[n].values() )
        ngram_maxcount.append( ( n, maxcount ) )
    return ngram_maxcount
    

def get_largest_idea_size( n_ngram_counts, min_size=1 ):
    ngram_maxcount = get_ngram_maxcount( n_ngram_counts )
    deltas = []
    for i in range( min_size, len( ngram_maxcount ) ):
        deltas.append( ( ngram_maxcount[i][0], ngram_maxcount[i][1] - ngram_maxcount[i-1][1] ) )

    largest_decrease = min( deltas, key=lambda x:x[1] )
    largest_idea_size = largest_decrease[0] - 1
    return largest_idea_size


def get_new_partitions_assignments( partition_start, partition_end, partition_wordids, max_idea, idea_idx ):
    assert len( partition_wordids ) == ( partition_end - partition_start )
    boundary_indices = [ partition_start ]
    assignments = []
    n = len( max_idea )
    assert n <= len( partition_wordids )

    part_idx = 0
    while part_idx < len( partition_wordids ) - n + 1:
        #print part_idx
        if tuple( partition_wordids[part_idx:part_idx+n] ) == tuple( max_idea ):
            start = partition_start + part_idx
            end = partition_start + part_idx + n
            if part_idx == 0:
                boundary_indices.append( end )
                assignments.append( idea_idx )
            else:
                boundary_indices.append( start )
                boundary_indices.append( end )
                assignments.append( None )
                assignments.append( idea_idx )
            part_idx += n
        else:
            part_idx += 1


            
    if part_idx != len( partition_wordids ):
        boundary_indices.append( partition_end )
        assignments.append( None )

    # generate tuples
    boundary_tuples = [ (boundary_indices[i],boundary_indices[i+1]) for i in range(len( boundary_indices ) - 1 )]
    #print assignments
    #print boundary_tuples
    assert len( boundary_tuples ) == len( assignments )

    new_partition_assignments = zip( boundary_tuples, assignments )

    return new_partition_assignments
    
