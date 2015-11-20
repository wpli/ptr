import align
import metrics
import sys
sys.path.append('../')
import _0020_match_ideas

def find_new_partition_assignments(wordids_to_ideas, wordids, partitions_assignments_tuples, \
                                   ptr_data, new_ptr_params, pfst_params, alignment_library={}):
    new_partitions_assignments = {}
    for partition, assignment in partitions_assignments_tuples:
        partition_wordids = wordids[partition[0]:partition[1]]
        unigram_logprob = metrics.get_unigram_logprob(ptr_data, partition_wordids)
        unigram_logprob = metrics.get_unigram_logprob(ptr_data, partition_wordids)
        similar_ideas = _0020_match_ideas.find_similar_ideas(partition_wordids, wordids_to_ideas, new_ptr_params.ideas, num_ideas=5)
        matches = []
        for idx in similar_ideas:
            reference_idea = new_ptr_params.ideas[idx]
            ref_word_set = frozenset([reference_idea, partition_wordids])
            if ref_word_set in alignment_library:
                align_logprob = alignment_library[ref_word_set]
            else:
                alpha_matrix = align.forward_evaluate_log(reference_idea, partition_wordids, pfst_params)
                align_logprob = alpha_matrix[-1,-1]
                alignment_library[ref_word_set] = align_logprob

            if align_logprob > unigram_logprob:
                matches.append((align_logprob, idx))

        if len(matches) > 0:
            max_idea = max(matches, key=lambda x:x[0])[1]
            if new_ptr_params.ideas[max_idea] != partition_wordids:
                print ptr_data.get_string(new_ptr_params.ideas[max_idea])
                print ptr_data.get_string(partition_wordids)
                print
        else:
            max_idea = None
        new_partitions_assignments[partition] = max_idea
    return new_partitions_assignments, alignment_library
