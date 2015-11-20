import PTRExperiment
import sys
def get_initial(sents_dict):

    ptr_data = PTRExperiment.PTRData()
    ptr_params = PTRExperiment.PTRParameters()
    ct = 0
    for docid, sents_list in sents_dict.iteritems():
        ct += 1
        if ct % 100 == 0:
            sys.stderr.write( "%s " % ct )
        current_idx = 0
        partitions = []
        doc_word_list = []
        for sent in sents_list:
            words = sent.lower().split()
            partitions.append((current_idx, current_idx+len(words)))
            current_idx += len(words)
            doc_word_list += words
        #print partitions
        #print len(doc_word_list)
        if len(doc_word_list) > 0:
            ptr_data.add_doc(docid, doc_word_list)
            ptr_params.add_doc(docid, partitions)

    return ptr_data, ptr_params
