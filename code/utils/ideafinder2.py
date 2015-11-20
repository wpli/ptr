from __future__ import absolute_import
import collections
import math
import multiprocessing
import random
import sys

import utils.PTRExperiment as PTRExperiment
from utils.datastructures import SlidingBagOfWords
from utils.align import AlignScorer

LENGTH_MULTIPLIER = 1.5

# IMPORTANT the data object is global for easy sharing in multiprocessing
# hence, do not use more than one instance of IdeaFinder at once
data = None


def _get_wordids(doc_id, index, loffset, roffset):
    """
    Returns the wordids from the specified document with
    positions [index-loffset : index+roffset]
    It should still work even if the offsets are too high.
    """

    loffset = int(math.floor(loffset))
    roffset = int(math.ceil(roffset))
    a = max(0, index - loffset)
    b = min(len(data.docid_wordids[doc_id]), index + roffset)
    return data.docid_wordids[doc_id][a:b]


def local_alignment(a, b, match=1.0, mismatch=-1.0):
    """
    Local alignment with positive score for matches and negative for
    mismatches. In finds the area in string b that generate the highest
    matching score with string a. Any area at both ends of string b that is
    not matched will not contribute to the mismatch score.

    Returns (score, start_index, end_index). The indices are from string b.
    """

    # the dynamic programming state is (score, - starting_index)
    # we use the negative of the index so that when we maximise,
    # we pick the one that started earlier (assuming equal scores).

    dp = [(None, None)] * (len(b) + 1)

    for i in range(len(a)):
        ndp = [(None, None)] * (len(b) + 1)
        
        for j in range(1, len(dp)):
            if dp[j - 1][0] is not None: # substitution
                ndp[j] = (dp[j - 1][0] + mismatch, dp[j - 1][1])

            if dp[j][0] is not None: # deletion
                ndp[j] = max(ndp[j], (dp[j][0] + mismatch, dp[j][1]))

            if ndp[j - 1][0] is not None: # insertion
                ndp[j] = max(ndp[j], (ndp[j - 1][0] + mismatch, ndp[j - 1][1]))

            if a[i] == b[j - 1]: # match
                ndp[j] = max(ndp[j], (mismatch * i + match,
                                      -j))  # first match of the sequences
                if dp[j - 1][0] is not None: # continue sequence
                    ndp[j] = max(ndp[j], (dp[j - 1][0] + match, dp[j - 1][1]))

        dp = ndp

    elem = max(dp)
    if not elem[1]:  # complete mismatch
        return (mismatch * len(a), 0, 0)
    else:
        return (elem[0], -elem[1] - 1, dp.index(elem))

def local_alignment_logprob(a, b, align_scorer=None):
    """
    Local alignment with language model scores for matches and mismatches.
    In finds the area in string b that generate the highest
    matching score with string a. Any area at both ends of string b that is
    not matched will not contribute to the mismatch score.

    Returns (score, start_index, end_index). The indices are from string b.
    """

    # the dynamic programming state is (score, - starting_index)
    # we use the negative of the index so that when we maximise,
    # we pick the one that started earlier (assuming equal scores).
    
    if align_scorer == None:
        align_scorer = AlignScorer()
        score = align_scorer.score_linear
    
    dp = [(None, None)] * (len(b) + 1)

    for i in range(len(a)):
        ndp = [(None, None)] * (len(b) + 1)
        
        for j in range(1, len(dp)):
            if dp[j - 1][0] is not None: # substitution
                ndp[j] = (dp[j - 1][0] + score('SUB', b[j-1]), dp[j - 1][1])

            if dp[j][0] is not None: # deletion
                ndp[j] = max(ndp[j], (dp[j][0] + score('DEL', -1), dp[j][1]))

            if ndp[j - 1][0] is not None: # insertion
                ndp[j] = max(ndp[j], (ndp[j - 1][0] + score('ADD', b[j-1]), ndp[j - 1][1]))

            if a[i] == b[j - 1]: # match
                ndp[j] = max(ndp[j], (score('DEL',-1) * i + score('MATCH', b[j-1]),
                                      -j))  # first match of the sequences
                if dp[j - 1][0] is not None: # continue sequence
                    ndp[j] = max(ndp[j], (dp[j - 1][0] + score('MATCH', b[j-1]), \
                                          dp[j - 1][1]))

        dp = ndp

    elem = max(dp)
    if not elem[1]:  # complete mismatch
        return (score('DEL', -1) * len(a), 0, 0)
    else:
        return (elem[0], -elem[1] - 1, dp.index(elem))

    
def score_idea(positions_d, pos, loffset, roffset):
    """
    Given a dictionary of positions {doc_id: [position_indices...]} and a
    proposed idea given by pos and the offsets, score that idea by
    summing the scores of local alignment between the proposed idea and the
    other positions.

    The scoring function only takes into account a maximum of one position
    from each document while completely excluding the document from which
    the proposed idea originated.
    """

    idea = _get_wordids(pos[0], pos[1], loffset, roffset)
    total_score = 0

    for docid, positions in positions_d.items():
        doc_score = None
        if docid != pos[0]:
            for p in positions:
                doc_score = max(doc_score, local_alignment(
                    idea, _get_wordids(docid, p, LENGTH_MULTIPLIER * loffset,
                                       LENGTH_MULTIPLIER * roffset))[0], \
                                align_scorer=None)
            if doc_score:
                total_score += doc_score

    return total_score


def _increment_offsets(positions_d, pos, loffset, roffset, mlof, mrof,
                       increment,
                       prev_score=None):

    if prev_score is None:
        prev_score = score_idea(positions_d, pos, loffset, roffset)

    def increment_offset(prev_score, offset, min, max, scoring):
        i = increment
        while True:
            if i < 0 and offset + i < min:
                i = min - offset
            elif i > 0 and offset + i > max:
                i = max - offset

            if not i:
                break

            score = scoring(offset + i)
            if score > prev_score:
                offset += i
                prev_score = score
            else:
                break

        return prev_score, offset

    prev_score, roffset = increment_offset(
        prev_score, roffset, 1, mrof,
        lambda x: score_idea(positions_d, pos, loffset, x))

    prev_score, loffset = increment_offset(
        prev_score, loffset, 0, mlof,
        lambda x: score_idea(positions_d, pos, x, roffset))

    return (prev_score, loffset, roffset)


def _find_best_offsets(inp):
    # mlof: max left offset
    # mrof: max right offset
    N, positions_d, pos, mlof, mrof = inp
    score, loffset, roffset = _increment_offsets(positions_d, pos, 0, N, mlof,
                                                 mrof, N)
    score, loffset, roffset = _increment_offsets(positions_d, pos, loffset,
                                                 roffset, mlof, mrof, -1,
                                                 score)
    score, loffset, roffset = _increment_offsets(positions_d, pos, loffset,
                                                 roffset, mlof, mrof, 1, score)
    return score, loffset, roffset, pos


class NGramCounter():
    """
    A class to keep track of the counts of ngrams in the data.
    It automatically gets rid of ngrams that occur less than two times
    to free up memory and speed up sorting.
    """

    def __init__(self, data, N):
        self.all = collections.defaultdict(list)
        self.deleted = collections.defaultdict(list)
        self.N = N
        self.data = data
        self._pre_process()

    def _pre_process(self):
        for docid, wordids in self.data.docid_wordids.items():
            for i in range(max(0, len(wordids) - self.N + 1)):
                ngram = tuple(wordids[i:i + self.N])
                self.all[ngram].append((docid, i))

        for ngram in self.all.keys():
            if len(self.all[ngram]) == 1:
                del self.all[ngram]

    def remove(self, ngram, position):
        if ngram not in self.all:
            return
        elif len(self.all[ngram]) - len(self.deleted[ngram]) <= 2:
            del self.all[ngram]
            del self.deleted[ngram]
        else:
            self.deleted[ngram].append(position)

    def remove_text(self, docid, start, end):
        start = max(start, 0)
        end = min(end, len(self.data.docid_wordids[docid]))
        wordids = self.data.docid_wordids[docid][start:end]
        for i in range(len(wordids) - self.N + 1):
            self.remove(wordids[i:i + self.N], (docid, start + i))

    def remove_all(self, ngram):
        try:
            del self.all[ngram]
        except KeyError:
            pass

        try:
            del self.deleted[ngram]
        except KeyError:
            pass

    def get_positions(self, ngram):
        if ngram not in self.all:
            return []

        self.all[ngram].sort()
        deleted = self.deleted[ngram]
        deleted.sort()
        positions = []
        i = 0

        for pos in self.all[ngram]:
            while i != len(deleted) and deleted[i] < pos:
                i += 1
            if i == len(deleted) or deleted[i] > pos:
                positions.append(pos)

        return positions

    def get_sorted_relevant_ngrams(self):
        ngrams = self.all.keys()
        ngrams.sort(key=lambda x: len(self.all[x]) - len(self.deleted[x]),
                    reverse=True)
        return ngrams


class IdeaExtractor():
    def __init__(self, data, N=5):
        self.data = data
        self.N = N
        self.clear()

    def clear(self):
        # wordids of extracted ideas
        self.ideas = []
        # dictionary of ngrams in assigned partitions
        self.ideas_ngrams = NGramCounter(self.data, self.N)
        # count of the number of times the idea appears in the corpus
        self.ideas_counts = []
        # positions of ideas in each doc: (start, end, idea_index)
        self.ideas_per_doc = collections.defaultdict(list)

    def extract_idea(self, idea):
        """
        Given a proposed idea, find it in the corpus, add it to the list of
        discovered ideas, mark all positions in which it appears,
        and save its ngrams.
        """

        idea_index = len(self.ideas)
        match_length = int(math.ceil(len(idea) * LENGTH_MULTIPLIER))
        bag_of_words = SlidingBagOfWords(idea, match_length, len(idea) / 2)
        count = 0

        # doc_count = 0
        for docid, wordids in self.data.docid_wordids.items():
            # doc_count += 1
            # sys.stderr.write("%s %s..." % (doc_count, len(wordids)))
            # sys.stderr.flush()
            # if doc_count % 100 == 0:
            #     sys.stderr.write("%s " % doc_count)

            a, b = 0, 0  # start and end of sliding window
            doc_ideas = self.ideas_per_doc[docid]
            i = 0  # index in doc_ideas
            bag_of_words.use_doc(wordids)

            while bag_of_words.next_match() is not None:
                if i < len(doc_ideas):
                    max_b = doc_ideas[i][0]
                else:
                    max_b = len(wordids)

                a = bag_of_words.next_match()
                b = min(max_b, a + match_length)

                if b - a < len(idea) / 2:
                    if i < len(doc_ideas):
                        bag_of_words.slide_to(doc_ideas[i][1])
                        i += 1
                        continue
                    else:
                        break

                (score, start, end) = local_alignment_logprob(idea, wordids[a:b], \
                                                              self.align_scorer)

                while start != 0 and b < max_b:
                    a += start
                    b = min(max_b, a + match_length)
                    (score, start, end) = local_alignment_logprob(idea, \
                                                                  wordids[a:b], \
                                                                  self.align_scorer)

                if end != 0: # different criteria:  # TODO find a better critieria
                    doc_ideas.insert(i, (a + start, a + end, idea_index))
                    self.ideas_ngrams.remove_text(
                        docid, a + start - self.N + 1, a + end + self.N - 1)
                    count += 1
                    i += 1
                    bag_of_words.slide_to(a + end)
                else:
                    bag_of_words.slide()

        self.ideas.append(idea)
        self.ideas_counts.append(count)

    def generate_ptr_params(self):
        params = PTRExperiment.PTRParameters()
        idea_bank = PTRExperiment.IdeaBank()
        total_bg_partitions = 0

        partitions = collections.defaultdict(dict)

        for docid, parts in self.ideas_per_doc.items():
            a, i = 0, 0
            while i < len(parts):
                if a != parts[i][0]:
                    partitions[docid][(a, parts[i][0])] = None
                    a = parts[i][0]
                    total_bg_partitions += 1
                else:
                    partitions[docid][parts[i][0:2]] = parts[i][2]
                    a = parts[i][1]
                    i += 1

            if a != len(self.data.docid_wordids[docid]):
                partitions[docid][(a, len(self.data.docid_wordids[docid]))
                                  ] = None
                total_bg_partitions += 1

        n_partitions = sum(self.ideas_counts) + total_bg_partitions

        for i in range(len(self.ideas)):
            idea_bank.add_idea(tuple(self.ideas[i]), None,
                               float(self.ideas_counts[i]) / n_partitions)

        params.ideas = idea_bank
        params.docid_partitions = partitions

        return params

    def generate_params_from_ideas(self, ideas):
        for idea in ideas:
            self.extract_idea(idea)
        return self.generate_ptr_params()


class IdeaFinder():
    """
    A class to find ideas in a set of documents that generate parameters with
    a high probability from the Probabilistic Text Reuse model.
    """

    def __init__(self, ptr_data,
                 ptr_params = None,
                 ngram_size=5,
                 scoring_iterations=16,
                 min_idea_length=7,
                 max_offset_length=None):
        global data
        data = ptr_data

        self.wordid_count = ptr_data.wordid_count
        if ptr_params != None:
            self.action_prob = ptr_params.action_prob
        else:
            self.action_prob = { 'MATCH': 0.8, 'ADD': 0.1, \
                                 'SUB': 0.1 }

        self.align_scorer = AlignScorer(self.action_prob, \
                                              self.wordid_count)
            
        self.data = ptr_data
        assert ngram_size <= min_idea_length
        self.N = ngram_size
        self.min_idea_length = min_idea_length
        self.max_offset_length = max_offset_length
        # number of iterations required to be satisfied with a maximum score
        self.scoring_iterations = scoring_iterations
        self._pool = multiprocessing.Pool()
        self.clear()

    def clear(self):
        self.extractor = IdeaExtractor(self.data, self.N)

    def get_wordids(self, doc_id, index, loffset, roffset):
        """
        Returns the wordids from the specified document with
        positions [index-loffset : index+roffset]
        It should still work even if the offsets are too high.
        """

        loffset = int(math.floor(loffset))
        roffset = int(math.ceil(roffset))
        a = max(0, index - loffset)
        b = min(len(self.data.docid_wordids[doc_id]), index + roffset)
        return self.data.docid_wordids[doc_id][a:b]

    def _get_max_offsets(self, docid, pos):
        """
        Returns the maximum possible offsets for a given position
        by ensuring that the offsets will not cross into the
        territory of an already assigned partition.
        """

        partitions = self.extractor.ideas_per_doc[docid]
        a, b = 0, len(partitions) - 1

        if b == -1:
            return (pos, len(self.data.docid_wordids[docid]) - pos)

        while a < b:
            mid = a + (b - a) / 2

            if partitions[mid][0] > pos:
                b = mid
            else:
                a = mid + 1

        if partitions[b][0] > pos:
            start = 0 if not b else partitions[b - 1][1]
            end = partitions[b][0]
        else:
            start = partitions[-1][1]
            end = len(self.data.docid_wordids[docid])

        if start > pos:
            raise ValueError('given position is already a part of a partition')

        return (pos - start, end - pos)

    def _find_best_idea_from_ngram(self, ngram, positions):
        """
        Find the idea with the (approximately) highest score using the provided
        ngram as its core.
        """

        best_result = None
        best_score = None
        positions_d = collections.defaultdict(list)

        for docid, position in positions:
            positions_d[docid].append(position)

        while True:
            inputs = []
            for j in range(self.scoring_iterations):
                pos = positions[random.randint(0, len(positions) - 1)]
                max_loffset, max_roffset = self._get_max_offsets(*pos)

                inputs.append((self.N, positions_d, pos, max_loffset,
                               max_roffset))

            outputs = self._pool.map(_find_best_offsets, inputs)
            
            result = max(outputs)

            if result[0] > best_score:
                best_result = result
                best_score = result[0]
            else:
                break

        pos, loffset, roffset = best_result[3], best_result[1], best_result[2]
        return self.get_wordids(pos[0], pos[1], loffset, roffset)

    def _find_next_idea(self):
        ngrams = self.extractor.ideas_ngrams.get_sorted_relevant_ngrams()
        for ngram in ngrams:
            sys.stderr.write("Finding idea from %s..." %
                             data.get_words_from_wordids(ngram))
            sys.stderr.flush()
            
            idea = self._find_best_idea_from_ngram(
                ngram, self.extractor.ideas_ngrams.get_positions(ngram))

            sys.stderr.write("Considering %s..." %
                             data.get_words_from_wordids(idea))
            sys.stderr.flush()

            
            if len(idea) < self.min_idea_length:
                self.extractor.ideas_ngrams.remove_all(ngram)
            else:
                self.extractor.extract_idea(idea)
                sys.stderr.write("done.\nExtracted (%d): %s\n\n" %
                                 (self.extractor.ideas_counts[-1],
                                  ' '.join(data.get_words_from_wordids(idea))))
                sys.stderr.flush()
                return self.extractor.generate_ptr_params()

        return None

    def find_optimal_parameters_by_iteration(self):
        new_ptr_params = self._find_next_idea()
        return new_ptr_params
