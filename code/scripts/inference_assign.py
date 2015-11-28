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
    with open(os.path.join(data_directory, 'ptr_params.greedy.001.p')) as f:
        ptr_params = cPickle.load(f)

    pfst_params = PTRExperiment.PFSTParams(ptr_data, ptr_params)

    logging.info('Loading assignment library')
    with open(os.path.join(data_directory, 'ptr_assignment_library.greedy.p')) as f:
        assignment_library = cPickle.load(f)

    logging.info('Loading alignment library')
    with open(os.path.join(data_directory, 'ptr_alignment_library.greedy.p')) as f:
        alignment_library = cPickle.load(f)


    # Assignment
    logging.info('Starting assignment')
    doc_ct = 0
    for docid, wordids in ptr_data.docid_wordids.iteritems():
        doc_ct += 1
        if doc_ct % 1000 == 0:
            logging.info('%s sentences done, current count: %s' % (sentence_ct, count))

        for p, assignment in ptr_params.docid_partitions[docid].iteritems():
            assert assignment == None
            partition_wordids = tuple(wordids[p[0]:p[1]])
            ptr_params.docid_partitions[p] = assignment_library.get(partition_wordids, None)

    logging.info('Writing new_ptr_params')
    with open(os.path.join(data_directory, 'ptr_params.greedy.002.p'), 'w') as f:
        cPickle.dump(ptr_params, f)

if __name__ == '__main__':
    main()
