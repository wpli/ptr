from __future__ import absolute_import

import collections
import utils.align as align
import numpy

class PTRData:
    def __init__(self):
        self.docid_wordids = {}
        self.wordid_count = collections.defaultdict(int)
        self.num_docs = 0
        self.num_tokens = 0
        self.num_words = 0
        self.word_wordid = {}
        self.wordid_word = {}
        self.vocab = []

    def add_doc(self, docid, words):
        wordids = tuple([self._get_wordid(i) for i in words])
        assert docid not in self.docid_wordids
        self.docid_wordids[docid] = wordids
        self.num_docs += 1
        assert self.num_docs == len(self.docid_wordids)

        for wordid in wordids:
            self.wordid_count[wordid] += 1

        self.num_tokens += len(wordids)

    def _get_wordid(self, word):
        if word not in self.word_wordid:
            self.word_wordid[word] = len(self.vocab)
            self.wordid_word[len(self.vocab)] = word
            self.vocab.append(word)
            self.num_words += 1
        return self.word_wordid[word]

    def get_wordids_from_words(self, words):
        return [self.word_wordid[i] for i in words]

    def get_words_from_wordids(self, word_ids):
        return [self.wordid_word[i] for i in word_ids]

    def get_string(self, word_ids):
        return " ".join(self.get_words_from_wordids(word_ids))
    
    def get_unigram_logprob(self, wordid):
        if wordid in self.wordid_count:
            frac = float(self.wordid_count[wordid]) / self.num_tokens
        elif wordid == "<DEL>":
            frac = 1.0 / self.num_tokens
        else:
            frac = 1.0 / self.num_tokens

        return numpy.log(frac)

    def get_idea_logprob(self, ref_wordids, query_wordids,
                         match=0.8,
                         mismatch=0.1):
        x = align.AlignedPassages(ref_wordids, query_wordids)
        x.global_align()
        logprob = 0.0

        for idx, i in enumerate(x.alignment_operations):
            if i == "MATCH":
                logprob += numpy.log(match)
            elif i in ("ADD", "SUB"):
                logprob += numpy.log(mismatch)

                logprob += self.get_unigram_logprob(
                    x.aligned_query_passage[idx])
            elif i == "DEL":
                logprob += numpy.log(mismatch)
                logprob += self.get_unigram_logprob("<DEL>")

        return logprob

    def get_background_logprob(self, query):
        logprob = 0.0
        for w in query:
            logprob += self.get_unigram_logprob(w)

        return logprob

class PTRParameters:
    def __init__(self, ptr_data=None):
        self.name = None
        self.docid_partitions = {}
        self.ideas = IdeaBank()
        self.max_partition_length = 40
        self.precompute_dict = {}

        if ptr_data == None:
            pass
        else:
            self.create_baseline_parameters(ptr_data)

        self.action_prob = dict(zip(["MATCH", "SUB", "ADD"], [0.8, 0.1,
                                                              0.1]))

    def add_doc(self, docid, partitions):
        self.docid_partitions[docid] = {}
        for p in partitions:
            self.docid_partitions[docid][p] = None
        
    def create_baseline_parameters(self, ptr_data):
        for docid, wordids in ptr_data.docid_wordids.iteritems():
            self.docid_partitions[docid] = {(0, len(wordids)): None}

    def deactivate_idea_by_index(self, idx):
        self.ideas.deactivate_idea(idx)
        for docid, partitions in self.docid_partitions.iteritems():
            for partition, assignment in partitions.iteritems():
                if assignment == idx:
                    self.docid_partitions[docid][partition] = None


class IdeaBank(dict):
    def __init__(self):
        self.active_ideas = set()

        self.inactive_ideas = set()
        self.wordids_idx_dict = {}
        self.idea_prob = {None: 1.0}

    def add_idea(self, word_ids, idea_idx=None, prob=0.001):
        tuple_word_ids = tuple(word_ids)

        if idea_idx == None:
            idea_idx = len(self)

        assert idea_idx not in self
        self[idea_idx] = tuple_word_ids
        self.idea_prob[idea_idx] = prob
        self.idea_prob[None] -= prob
        assert self.idea_prob[None] >= 0.0
        self.active_ideas.add(idea_idx)
        self.wordids_idx_dict[word_ids] = idea_idx

        return idea_idx

    def deactivate_idea(self, idx):
        self.active_ideas.remove(idx)
        self.inactive_ideas.add(idx)
        self.idea_prob[None] += self.idea_prob[idx]
        del self.idea_prob[idx]
        assert len(set.intersection(self.active_ideas, self.inactive_ideas)) == 0
        assert len(set.union(self.active_ideas,
                             self.inactive_ideas)) == len(self)

    def get_active_idea_set(self):
        return self.active_ideas

class PFSTParams:
    def __init__(self, ptr_data, ptr_params):
        self.word_wordid = ptr_data.word_wordid
        self.wordid_word = ptr_data.wordid_word
        self.wordid_count = ptr_data.wordid_count 
        self.num_words = len(ptr_data.word_wordid)
        self.num_tokens = float(ptr_data.num_tokens)
        self.trans_dict = collections.defaultdict(int)
   
    def trans(self, source, sink):
        return self.initial_trans(source, sink)
    
        #if self.trans_dict == {}:
        #    
        #else:
        #    return self.learned_trans(source, sink)
           
    def initial_trans(self, source, sink):
        if source == sink: # match
            prob = 0.8
        elif source == None: # deletion
            prob = 0.1 * 1.0 / (self.num_words+1)
        elif sink == None: # addition
            prob = 0.1 * 1.0 / (self.num_words+1)
        else:
            prob = 0.1 * 1.0 / self.num_words # mismatch
        return prob
    
    def learned_trans(self, source, sink):
        return self.trans_dict[source, sink]
    
