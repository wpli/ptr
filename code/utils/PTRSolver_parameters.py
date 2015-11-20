def get_new2( new_partitions_assignments, partition, partition_wordids, current_idea, current_idea_idx ):

    partition_start = partition[0]
    partition_end = partition[1]
    assert len( partition_wordids ) == ( partition_end - partition_start )
    boundary_indices = [ partition_start ]
    assignments = []
    n = len( current_idea )
    assert n <= len( partition_wordids )

    part_idx = 0
    while part_idx < len( partition_wordids ) - n + 1:
        #print part_idx
        if tuple( partition_wordids[part_idx:part_idx+n] ) == tuple( current_idea ):
            new_dict = {}
            start = partition_start + part_idx
            end = partition_start + part_idx + n
            new_dict = parameter_operations.reassign_partition( new_partitions_assignments, partition, current_idea_idx, len( current_idea ), offset=part_idx )
            
            #if part_idx == 0:
            #    boundary_indices.append( end )
            #    assignments.append( idea_idx )
            #else:
            #    boundary_indices.append( start )
            #    boundary_indices.append( end )
            #    assignments.append( None )
            #    assignments.append( idea_idx )
            tups = new_dict.tuples()
            boundary_indices += [ i[0] for i in tups ]
            assignments += [ i[1] for i in tups ]
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




