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

def get_text_sentences(filepath, sbd_model):
    tokens_by_sentence = []
    with codecs.open(filepath, encoding='utf8') as f:
        raw_text = f.read()
        #raw_text = raw_text.lower()
        raw_text = strip_multiple_whitespaces(raw_text)
        sentences = splitta.sbd.sbd_text(sbd_model, raw_text, do_tok=False)
        for s in sentences:
            new_s = strip_punctuation(s)
            tokens_by_sentence.append(list(utils.tokenize(new_s, deacc=True, lowercase=True)))
        #print raw_text
        #for filt in self.preprocess:
        #    raw_text = filt(raw_text)
        #text = list(utils.tokenize(raw_text, deacc=True, lowercase=True))
    return sentences, tokens_by_sentence

def main():
    data_path = '../../data/fcc/sunlight_full_partitions/'
    train_directories = [str(i) for i in range(8)]



    sbd_model = splitta.sbd.load_sbd_model('../utils/splitta/model_nb/', False)
    
    filepaths = []
    for td in train_directories:
        for fname in os.listdir(os.path.join(data_path, td)):
            filepaths.append(os.path.join(data_path, td, fname))
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences_dict = {}
    tokens_by_sentence_dict = {}
    for idx, filepath in enumerate(filepaths):
        if idx % 1000 == 0:
            sys.stderr.write("%s " % idx)
        sentences, tokens_by_sentence = get_text_sentences(filepath, sbd_model)
        sentences_dict[filepath] = sentences
        tokens_by_sentence_dict[filepath] = tokens_by_sentence
        #print sentences
        #print tokens_by_sentence

    # filters
    #filters = [strip_punctuation, strip_multiple_whitespaces] #, strip_numeric, remove_stopwords, strip_short]

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
