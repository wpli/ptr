import sys
sys.path.append('../')
from utils.utils_gensim import FolderCorpus
import codecs
import os, logging, sys
import utils.splitta as splitta
import gensim
from gensim import corpora, models, utils
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, STOPWORDS
import cPickle
from utils import PTRExperiment

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('Loading sentences dict...')

    
    sentences_dict_file = '../../experiments/ptr/sunlight_full_train.sentences_dict.p'
    with open(sentences_dict_file) as f:
        sentences_dict = cPickle.load(f)

    logging.info('Loading files')
    ct = 0
    tokens_by_sentence_dict = {}
    for fname, sentence_list in sentences_dict.iteritems():
        ct += 1
        tokens_by_sentence = []
        for s in sentence_list:
            new_s = strip_punctuation(s)
            tokens_by_sentence.append(list(utils.tokenize(new_s, deacc=True, lowercase=True)))
        tokens_by_sentence_dict[fname] = tokens_by_sentence
        if ct % 10000 == 0:
            logging.info('Writing tokens by sentence %s' % ct)
            with open('../../experiments/ptr/sunlight_full_train.tokens_by_sentence.%s.p' % str(ct), 'w') as f:
                cPickle.dump(tokens_by_sentence_dict, f)
            tokens_by_sentence_dict = {}

    sys.exit()
    #data_path = '../../data/fcc/sunlight_full_partitions/'

    # create corpus
    output_path = '../../experiments/ptr/'
    filename = 'sunlight_full_train'

    #logger.info("Saving files")
    with open(os.path.join(output_path, filename + ".sentences_dict.p"), 'w') as f:
        cPickle.dump(sentences_dict, f)

    with open(os.path.join(output_path, filename + ".tokens_by_sentence_dict.p"), 'w') as f:
        cPickle.dump(tokens_by_sentence_dict, f)

    # write files
    #logger.info('Saving filenames to disk: {}.txt'.format(filename))
    #with open(os.path.join(output_path, filename +".txt"), 'w') as f:
    #    for fpath in filepaths:
    #        f.write(fpath + "\n")

    # ptr_data


    #MmCorpus.serialize('{}.mm'.format(outfile_path), corpus, progress_cnt=1000)
    #logger.info('Saving dictionary to disk: {}.dict'.format(filename))
    #corpus.dictionary.save('{}.dict'.format(outfile_path))
    

if __name__ == '__main__':
    main()
