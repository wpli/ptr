import align
import numpy
import sys
import collections
import PTRExperiment
import heapq

def wordids_to_words( sorted_vocab, word_ids ):
    return tuple( [ sorted_vocab[i] for i in word_ids ] ) 

def get_top_candidates(query_wordids, ptr_params, num_top_candidates=50, jaccard_cutoff=0.3):
    idea_hit_counter = collections.defaultdict(int)
    for wordid in query_wordids:
        idea_set = ptr_params.ideas.wordids_ideas_inverted_index[wordid]
        for i in list(idea_set):
            idea_hit_counter[i] += 1
    top_ideas = heapq.nlargest(num_top_candidates, idea_hit_counter.iteritems(), key=lambda x:x[1])

    top_candidates = []
    for cand, count in top_ideas:
        jacc = count / \
            float(len(set.union(set(ptr_params.ideas[cand]), set(query_wordids))))
        if jacc > jaccard_cutoff:
            top_candidates.append(cand)

    return top_candidates

def get_assignment(wordids, candidate_ideas, ptr_data, ptr_params, pfst_params):
    background_logprob = get_unigram_logprob(ptr_data, wordids)
    best_logprob = background_logprob
    best_idea = None

    for idea in candidate_ideas:
        idea_wordids = ptr_params.ideas[idea]
        idea_logprob = get_align_logprob(ptr_params.ideas[idea], wordids, pfst_params)
        if idea_logprob > best_logprob:
            best_logprob = idea_logprob
            best_idea = idea

    return best_idea
    

def get_sentence_boundaries( sorted_vocab, word_ids ):
    boundaries = [0,len( word_ids ) ]
    for idx, word_id in enumerate( word_ids ):
        if sorted_vocab[word_id][-1] == "." or \
           sorted_vocab[word_id][-1] == "?" \
           or sorted_vocab[word_id][-1] == "!":
            #new_word = sorted_vocab[word_id][:-1]
            #new_word_id = vocab_idx_dict[new_word]
            #new_word_ids[idx] = new_word_id
            boundaries.append( idx+1 )
        else:
            pass
        
    boundaries = list( set( boundaries ) )
    boundaries.sort()
    return boundaries

"""
#returns list of word tuples 
"""
def split_sentences( sorted_vocab, vocab_idx_dict, word_ids ):
    new_word_ids = list( word_ids )
    boundaries = [0,len( word_ids ) ]
    for idx, word_id in enumerate( word_ids ):

        if sorted_vocab[word_id][-1] == "." or sorted_vocab[word_id][-1] == "?":
            
            #new_word = sorted_vocab[word_id][:-1]
            #new_word_id = vocab_idx_dict[new_word]
            #new_word_ids[idx] = new_word_id
            boundaries.append( idx+1 )
        else:
            pass
        
    boundaries = list( set( boundaries ) )
    boundaries.sort()
    
    new_tuples = []
    
    for idx, i in enumerate( boundaries[:-1] ):
        start = boundaries[idx]
        end = boundaries[idx+1]
        new_tuples.append( tuple( new_word_ids[start:end] ) )
    
    return new_tuples

def get_unigram_logprob(ptr_data,word_ids):
    word_counter_dict = ptr_data.wordid_count
    total_tokens = ptr_data.num_tokens
    total_vocab = len(ptr_data.word_wordid)
    total_logprob = 0.0
    # Add-one smoothing
    if type(word_ids) == int:
        word_ids = [ word_ids ]
    
    for word_id in word_ids:
        if word_id in word_counter_dict:
            count = word_counter_dict[word_id]
        else: 
            count = 0
        total_logprob += numpy.log(count + 1) - numpy.log(total_tokens + \
                                                             total_vocab )
        #print float(count+1) / ( total_tokens+total_vocab)
        
    return total_logprob

def get_total_partitions(ptr_params):
    total_partitions = 0
    for docid, partitions in ptr_params.docid_partitions.iteritems():
        total_partitions += len(partitions)
    return total_partitions

def get_partition_coverage(ptr_params):
    total_partitions = get_total_partitions(ptr_params)
    total_none = get_idea_count(ptr_params,None)
    return 1.0 - float(total_none)/total_partitions

def get_word_coverage(ptr_params):
    total_idea_words = 0
    total_background_words = 0
    for docid,partitions_assignments in ptr_params.docid_partitions.iteritems():
        for partition, assignment in partitions_assignments.iteritems():
            if assignment == None:
                total_background_words += partition[1] - partition[0]
            else:
                total_idea_words += partition[1] - partition[0]

    # print total_idea_words, total_background_words
    
    return float(total_idea_words)/(total_idea_words+total_background_words)
            

def get_idea_count(ptr_params,idx):
    flat = []
    for i in ptr_params.docid_partitions.values():
        flat += i.values()
    idea_counts = collections.Counter(flat) 
    return idea_counts[idx]

def get_idea_logprob( word_ids, idea ):
    jaccard = calculate_jaccard( word_ids, idea )
    return numpy.log( jaccard )
    
    
def calculate_jaccard( candidate_idea, reference_idea ):
    candidate_ids = set( candidate_idea )
    reference_ids = set( reference_idea )
    num = float( len( set.intersection( candidate_ids, reference_ids ) ) )
    denom = len( set.union( candidate_ids, reference_ids ) )
    return num / denom

def print_sentence( sorted_vocab, word_ids ):
    return " ".join( [ sorted_vocab[i] for i in word_ids ] )

def jaccard( query_list, ref_list ):
    query_set = set( query_list )
    ref_set = set( ref_list )
    
    numerator = len( set.intersection( query_set, ref_set ) )
    denominator = len( set.union( query_set, ref_set ) )
    
    return float( numerator ) / float( denominator )


def compute_logprob_poisson( idea_length, lam=100):
    return idea_length * numpy.log( lam ) - lam - \
        sum( [ numpy.log(i) for i in range(1,idea_length+1)] )


def compute_corpus_logprob(ptr_data, ptr_params, pfst_params, \
                           alignment_library={}, \
                           verbose=True):
    
        
    #align_scorer = align.AlignScorer(ptr_params.action_prob, \
    #                                 ptr_data.wordid_count)
                                     
    logprob_total = 0.0
    # ideas
    logprob_ideas = 0.0
    for idea_id, idea in ptr_params.ideas.iteritems():
        #poisson_logprob = compute_logprob_poisson(len(idea), lam=30)
        idea_generation_logprob = numpy.log(1.0) - numpy.log(45.0)
        word_counter_dict = ptr_data.wordid_count
        total_tokens = ptr_data.num_tokens
        total_vocab = ptr_data.num_words
        
        idea_words_logprob = get_unigram_logprob(ptr_data, idea)
        logprob_ideas += idea_generation_logprob
        logprob_ideas += idea_words_logprob

    # partitions
    max_partitions = 1000
    logprob_partitions = 0.0
    for docid, partitions in ptr_params.docid_partitions.iteritems():
        if len( partitions ) <= max_partitions:
            pass
        else:
            if verbose:
                sys.stderr.write("Warning: %s has %s partitions\n" % \
                                 (docid,len(partitions)))
        logprob_partitions += numpy.log(1.0/max_partitions)
    
    # assignments
    logprob_assignments = 0.0
    
    for docid, partitions in ptr_params.docid_partitions.iteritems():
        for partition, assignment in partitions.iteritems():
            prob_idea = ptr_params.ideas.idea_prob[assignment]
            logprob_assignments += numpy.log( prob_idea )


    # \Pr(D|PKA) is the probability of the text 
    logprob_text = 0.0
    ct = 0
    for docid, partitions in ptr_params.docid_partitions.iteritems():
        if ct % 1000 == 0:
            sys.stderr.write("%s " % ct)
            sys.stderr.flush()

        ct += 1
        
        for partition, assignment in partitions.iteritems():
            logprob_passage = 0.0
            start_idx = partition[0]
            end_idx = partition[1]

            if assignment == None:
                logprob_passage = get_unigram_logprob(ptr_data, \
                                    ptr_data.docid_wordids[docid][start_idx:end_idx])
                #query_idea = tuple( ptr_data.docid_wordids[docid][start_idx:end_idx] )
                #ref_word_set = frozenset([None, tuple(query_idea)])
            else:
                # infer the probability of the passage
                ref_idea = ptr_params.ideas[assignment]
                query_idea = tuple( ptr_data.docid_wordids[docid][start_idx:end_idx] )
                ref_word_set = frozenset([tuple(ref_idea), tuple(query_idea)])
                if ref_word_set in alignment_library:
                    logprob_passage += alignment_library[ref_word_set]
                else:
                    logprob_passage = get_align_logprob(ref_idea, query_idea, \
                                                    pfst_params)
                    alignment_library[ref_word_set] = logprob_passage
                
                
            logprob_text += logprob_passage
    logprob_total = logprob_ideas + logprob_partitions + logprob_assignments \
                    + logprob_text
    
    return logprob_total

def get_align_logprob(ref_idea, query_idea, align_scorer=None):
    alpha_matrix = align.forward_evaluate_log(ref_idea, query_idea, align_scorer)
    logprob_passage = alpha_matrix[-1,-1]
    return logprob_passage
    
    #logprob_passage = 0.0
    
    #aligned_passages = align.AlignedPassages(ref_idea, query_idea, align_scorer)
    #aligned_passages.global_align()
    #logprob_passage = aligned_passages.final_score
    #ops = aligned_passages.alignment_operations
    #if len(aligned_passages.alignment_operations) != len(aligned_passages.query_passage):
    #    print aligned_passages.alignment_operations
    #    print aligned_passages.aligned_reference_passage
    #    print aligned_passages.aligned_query_passage
    #    print

    
    """
    for idx, op in enumerate(ops):
        if op == 'MATCH':
            logprob_passage += numpy.log(ptr_params.action_prob["MATCH"])
        elif op == 'SUB':
            logprob_passage += numpy.log(ptr_params.action_prob["SUB"]) + \
                               get_unigram_logprob(ptr_data,\
                                    aligned_passages.aligned_query_passage[idx])
            
        elif op == 'DEL':
            logprob_passage += numpy.log(ptr_params.action_prob["SUB"]) + \
                               get_unigram_logprob(ptr_data,-1)
        else:
            assert op == 'ADD'
            logprob_passage += numpy.log(ptr_params.action_prob["ADD"]) + \
                               get_unigram_logprob(ptr_data,\
                                    aligned_passages.aligned_query_passage[idx])
        #print op, logprob_passage
    """
    
    #return logprob_passage

def merge_ideas( ideas ):
    # find two most similar ideas
    pairs_jaccard = []
    for idx1, idea1 in ideas.items():
        for idx2, idea2 in ideas.items():
            if idx1 >= idx2:
                pass
            else:
                pairs_jaccard.append( ( (idx1,idx2), jaccard( idea1,idea2 ) ) )

    pairs_jaccard.sort( key=lambda x:x[1], reverse=True )
    return pairs_jaccard

def get_progress( counter, marker=10000 ):
    counter+=1
    if counter % marker == 0:
        sys.stderr.write( "%s " % counter )
    return counter

def correct_partitions( doc_partitions_assignments_dict ):
    new_doc_partitions_assignment_dict = {}
    ctr = 0
    for doc_id, word_ids in id_wordids:
        ctr = get_progress( ctr )
        partitions_assignments_dict = doc_partitions_assignments_dict[doc_id]
        if set( partitions_assignments_dict.values() ) == {None}:
            new_dict = collections.OrderedDict()
            new_dict[( 0, len( word_ids) )] = None
            doc_partitions_assignments_dict[doc_id] = new_dict
        elif len( partitions_assignments_dict ) == 1:
            pass
        else:
            new_dict = collections.OrderedDict()
            partitions_assignments = partitions_assignments_dict.items()
            partitions_assignments.sort( key=lambda x:x[0][0] )
            first_idx = None
            curr_idx = 0
            while curr_idx < len( partitions_assignments ):
                partition = partitions_assignments[curr_idx][0]
                assignment = partitions_assignments[curr_idx][1]
                if assignment == None:
                    if first_idx == None:
                        first_idx = curr_idx
                        start = partition[0]
                    else:
                        pass
                    curr_idx += 1
                else:
                    if first_idx == None:
                        new_dict[partition] = assignment
                        curr_idx += 1
                    else:
                        end = partition[0]
                        first_idx = None
                        new_dict[(start,end)] = None 
                        curr_idx += 1

            # if we reached the end and first_idx is not None, then we need to append the end
            if first_idx != None:
                end = len( word_ids )
                new_dict[(start,end)] = None
            new_doc_partitions_assignments_dict[doc_id] = new_dict
            
    return new_doc_partitions_assignment_dict

    

