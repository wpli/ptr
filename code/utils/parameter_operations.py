import PTRExperiment

"""
Reassign a partition and assign remainder of partition to background, if applicable.
"""
def reassign_partition( partitions_assignments_dict, target_partition, new_idea_idx, reassignment_length, offset=0 ):
    assert target_partition in partitions_assignments_dict
    assert target_partition[1] - target_partition[0] >= reassignment_length + offset

    if target_partition[1] - target_partition[0] == reassignment_length and offset == 0:
        partitions_assignments_dict[target_partition] = idea_idx
    elif offset == 0:
        del partitions_assignments_dict[target_partition]
        partitions_assignments_dict[ ( target_partition[0], target_partition[0] + reassignment_length ) ] = idea_idx
        partitions_assignments_dict[ ( target_partition[0] + reassignment_length ), target_partition[1] ] = None
    elif offset != 0:
        del partitions_assignments_dict[target_partition]
        partitions_assignments_dict[ ( target_partition[0], target_partition[0] + offset ) ] = None
        partitions_assignments_dict[ ( target_partition[0] + offset, target_partition[0] + offset + reassignment_length ) ] = idea_idx

        if target_partition[0] + offset + reassignment_length != target_partition[1]:
            partitions_assignments_dict[ ( target_partition[0] + offset + reassignment_length ), target_partition[1] ] = None
        
    return partitions_assignments_dict
        
    
    
        
    
    
        


