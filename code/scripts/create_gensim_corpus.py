import sys
sys.path.append('../')
from utils.utils_gensim import FolderCorpus
import codecs
import os, logging, sys
import gensim
from gensim import corpora, models, utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus 
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, STOPWORDS

def main():
    data_path = '../../data/fcc/sunlight_full_partitions/'
    train_directories = [str(i) for i in range(8)]

    filepaths = []
    for td in train_directories:
        for fname in os.listdir(os.path.join(data_path, td)):
            filepaths.append(os.path.join(data_path, td, fname))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('Archive.gensim')

    # filters
    filters = [strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]

    # create corpus
    output_path = '../../experiments/gensim/'
    filename = 'sunlight_full_train'

    logger.info('Creating list of filepaths...')
    logger.info('Creating corpus object...')
    corpus = FolderCorpus(filepaths=filepaths, preprocess=filters)

    outfile_path = os.path.join(output_path, filename)

    # write files
    logger.info('Saving filenames to disk: {}.txt'.format(filename))
    with open(os.path.join(output_path, filename +".txt"), 'w') as f:
        for fpath in filepaths:
            f.write(fpath + "\n")

    MmCorpus.serialize('{}.mm'.format(outfile_path), corpus, progress_cnt=1000)
    logger.info('Saving dictionary to disk: {}.dict'.format(filename))
    corpus.dictionary.save('{}.dict'.format(outfile_path))
    

if __name__ == '__main__':
    main()
