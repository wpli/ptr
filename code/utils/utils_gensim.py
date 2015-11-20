import codecs
import os, logging, sys
import gensim
from gensim import corpora, models, utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus 
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, STOPWORDS

class FolderCorpus(corpora.TextCorpus):
    def __init__(self, filepaths, preprocess=[], dictionary=None):
        self.filepaths = filepaths
        self.preprocess = preprocess
        self.metadata = None

        self.dictionary = Dictionary()

        self.dictionary.add_documents(self.get_texts())
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=500000)
        self.dictionary.compactify()

    def get_texts(self):
        for path in self.filepaths:
            with codecs.open(path, encoding='utf8') as f:
                raw_text = f.read()
                raw_text = raw_text.lower()
                for filt in self.preprocess:
                    raw_text = filt(raw_text)
                text = list(utils.tokenize(raw_text, deacc=True, lowercase=True))
                yield text
