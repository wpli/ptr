import sys
import os
sys.path.append('../')
from utils import metrics
from utils import PTRExperiment
import logging
import cPickle
import collections
import copy        

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    data_directory = '../../experiments/ptr/'

    # Load data
    logging.info('Loading ptr_data')
    with open(os.path.join(data_directory, 'ptr_data.p')) as f:
        ptr_data = cPickle.load(f)

    logging.info('Loading ptr_params')
    with open(os.path.join(data_directory, 'ptr_params.p')) as f:
        ptr_params = cPickle.load(f)

    pfst_params = PTRExperiment.PFSTParams(ptr_data, ptr_params)

    # Copy new parameters
    #logging.info('Creating new_ptr_params')
    #new_ptr_params = copy.deepcopy(ptr_params)
    #doc_ct = 0

    logging.info('Loading sentence counter')
    with open(os.path.join(data_directory, 'ptr_sentence_counter.p')) as f:
        sentence_counter = cPickle.load(f)

    sorted_sentence_counts = sorted(sentence_counter.items(), key=lambda x:x[1], reverse=True)
    del sentence_counter

    alignment_library = {}
    assignment_library = {}

    #frozenset([tuple(ref_idea), tuple(query_idea)])
    
    # Inference
    logging.info('Starting inference')
    sentence_ct = 0
    for wordids, count in sorted_sentence_counts:
        sentence_ct += 1
        if sentence_ct % 100 == 0:
            logging.info('%s sentences done, current count: %s' % (sentence_ct, count))
        
        if wordids in assignment_library:
            pass
        elif wordids in ptr_params.ideas.wordids_idx_dict:
            assignment_library[wordids] = ptr_params.ideas.wordids_idx_dict[wordids]
        else:
            top_ideas = metrics.get_top_candidates(sent_wordids, new_ptr_params, num_top_candidates=20, jaccard_cutoff=0.5)
            assignment = metrics.get_assignment(sent_wordids, top_ideas, ptr_data, new_ptr_params, pfst_params)
            assignment_library[wordids] = assignment

    logging.info('Writing new_ptr_params')
    with open(os.path.join(data_directory, 'ptr_assignment_library', 'w')) as f:
        cPickle.dump(assignment_library, f)

if __name__ == '__main__':
    main()
