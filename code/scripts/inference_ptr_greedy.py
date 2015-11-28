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
    logging.info('Creating new_ptr_params')
    new_ptr_params = copy.deepcopy(ptr_params)
    doc_ct = 0

    logging.info('Loading assignment and alignment libraries')
    with open(os.path.join(data_directory, 'ptr_assignment_library.p')) as f:
        assignment_library = cPickle.dump(assignment_library, f)

    with open(os.path.join(data_directory, 'ptr_alignment_library.p')) as f:
        alignment_library = cPickle.dump(alignment_library, f)


    # Inference
    logging.info('Starting inference')
    for docid, wordids in ptr_data.docid_wordids.iteritems():
        doc_ct += 1
        if doc_ct % 1000 == 0:
            logging.info('%s documents done' % doc_ct)
            #with open(os.path.join(data_directory, 'ptr_params.greedy.001.p', 'w')) as f:
            #    cPickle.dump(new_ptr_params, f)

        for p, assignment in ptr_params.docid_partitions[docid].iteritems():
            
            sent_wordids = tuple(wordids[p[0]:p[1]])

            if sent_wordids in new_ptr_params.ideas.wordids_idx_dict:
                new_ptr_params.docid_partitions[docid][p] = new_ptr_params.ideas.wordids_idx_dict[sent_wordids]
            else:
                top_ideas = metrics.get_top_candidates(sent_wordids, new_ptr_params, num_top_candidates=20, jaccard_cutoff=0.5)
                assignment = metrics.get_assignment(sent_wordids, top_ideas, ptr_data, new_ptr_params, pfst_params, alignment_library)
                new_ptr_params.docid_partitions[docid][p] = assignment

    logging.info('Writing new_ptr_params')
    with open(os.path.join(data_directory, 'ptr_params.greedy.001.p', 'w')) as f:
        cPickle.dump(new_ptr_params, f)


if __name__ == '__main__':
    main()
