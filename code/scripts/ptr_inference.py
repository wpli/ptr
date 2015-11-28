import sys
sys.path.append('../')

import codecs
import os, logging, sys
import utils.splitta as splitta
import cPickle
from utils import PTRExperiment
import utils import metrics

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    data_directory = '../../experiments/ptr/'

    logging.info('Loading ptr_data')
    with open(os.path.join(data_directory, 'ptr_data.p')) as f:
        ptr_data = cPickle.load(f)

    logging.info('Loading ptr_params')
    with open(os.path.join(data_directory, 'ptr_params.p')) as f:
        ptr_params = cPickle.load(f)

    logging.info('Loading ptr_params')

    sys.exit()



    files = [i for i in os.listdir('../../experiments/ptr/') if 'sunlight_full_train.tokens_by_sentence' in i]
    for fname in files:
        full_path = os.path.join(data_directory, fname)
        logging.info('Loading %s' % full_path)
        with open(full_path) as f:
            sentence_dict = cPickle.load(f)
            for fname, sentence_list in sentence_dict.iteritems():
                key = fname.split('/')[-1]
                combined = list(itertools.chain.from_iterable(sentence_list))
                sentence_lengths = [len(i) for i in sentence_list]
                ptr_data.add_doc(key, combined)
                
                prev_start = 0
                partitions = []
                for l in sentence_lengths:
                    partitions.append((prev_start, prev_start + l))
                    prev_start += l
                
                assert prev_start == len(combined)
                ptr_params.add_doc(key, partitions)
    
    logging.info('Writing ptr_data')
    with open(os.path.join(data_directory, 'ptr_data.p'), 'w') as f:
        cPickle.dump(ptr_data, f)

    logging.info('Writing ptr_params')
    with open(os.path.join(data_directory, 'ptr_params.p'), 'w') as f:
        cPickle.dump(ptr_params, f)

    

if __name__ == '__main__':
    main()
