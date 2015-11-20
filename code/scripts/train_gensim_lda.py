import logging, gensim

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('Archive.LDA')    
    # load dictionary and corpus
    dictionary = gensim.corpora.Dictionary.load('../../experiments/gensim/sunlight_full_train.dict')
    corpus = gensim.corpora.MmCorpus('../../experiments/gensim/sunlight_full_train.mm')

    # train LDA
    logger.info('Initiate training')
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, eval_every=10, chunksize=5000, passes=1)

    # save model
    model_path = '../../experiments/gensim/sunlight_full_train.lda'
    logger.info('Saving model to disk: {}'.format(model_path))
    model.save(model_path)

if __name__ == '__main__':
    main()
