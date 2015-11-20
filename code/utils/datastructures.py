import collections


class NGramNode():
    def __init__(self):
        self.edges = collections.defaultdict(NGramNode)  # wordid to ngramnode
        self.count = 0


class NGramTree():
    def __init__(self, max_depth):
        self.depth = max_depth
        self.root = NGramNode()

    def add_doc(self, docid, wordids):
        for i in range(len(wordids)):
            cur = self.root
            cur.count += 1
            for j in range(i, min(len(wordids), i + self.depth)):
                cur = cur.edges[wordids[j]]
                cur.count += 1

    def count(self, wordids):
        return self.count_with_deletions(wordids, 0)

    def counts_by_edit_distance(self, wordids):
        counts = [0] * (len(wordids) + 1)
        for i in range(len(counts)):
            counts[i] += self.count_with_deletions(wordids, i)
            for j in range(i + 1, len(counts)):
                if j == len(counts) - 1:
                    counts[j] -= counts[i] * (j - i)
                else:
                    counts[j] -= counts[i] * (j - i + 1)
            print i, counts
        return counts

    def count_with_deletions(self, wordids, deletions):
        if deletions > len(wordids):
            return 0

        return self._count_w_deletions(wordids, deletions, 0, self.root)

    def _count_w_deletions(self, wordids, deletions, index, node):
        if index >= len(wordids):
            if deletions == 0:
                return node.count
            return 0

        count = 0
        if deletions > 0:
            count += self._count_w_deletions(wordids, deletions - 1, index + 1,
                                             node)

        if wordids[index] in node.edges:
            count += self._count_w_deletions(wordids, deletions, index + 1,
                                             node.edges[wordids[index]])

        return count


class MaxHeap():
    def __init__(self):
        self.heap = []

    def add(self, elem, score):
        self.heap.append((elem, score))
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            return None
        else:
            (elem, score) = self.heap[0]
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            self.heap = self.heap[:-1]
            self._heapify_down(0)
            return (elem, score)

    def isempty(self):
        if len(self.heap) == 0:
            return True
        else:
            return False

    def _heapify_up(self, ind):
        if ind == 0:
            return

        parent = ind // 2
        if self.heap[parent][1] < self.heap[ind][1]:
            self.heap[parent], self.heap[ind] = self.heap[ind], self.heap[parent]
            self._heapify_up(parent)

    def _heapify_down(self, ind):
        child1 = ind * 2
        child2 = ind * 2 + 1

        if child1 >= len(self.heap):
            return
        elif child2 >= len(self.heap):
            child = child1
        else:
            if self.heap[child1][1] > self.heap[child2][1]:
                child = child1
            else:
                child = child2

        if self.heap[ind][1] < self.heap[child][1]:
            self.heap[ind], self.heap[child] = self.heap[child], self.heap[ind]
            self._heapify_down(child)


class SlidingBagOfWords():
    def __init__(self, idea, window_size, min_matches):
        assert window_size > 0
        self.idea = idea
        self.window_size = window_size
        self.min_matches = min_matches
        self.idea_unigrams = collections.defaultdict(int)

        for word in idea:
            self.idea_unigrams[word] += 1

    def use_doc(self, doc):
        self.doc = doc
        self.window_unigrams = collections.defaultdict(int)
        self.matches = 0
        self.i = 0

        for c in doc[:self.window_size]:
            self.window_unigrams[c] += 1

        for k, v in self.idea_unigrams.items():
            self.matches += min(v, self.window_unigrams[k])

        self.slide_to(0)

    def slide_to(self, start_position):
        i = self.i
        while i < len(self.doc) and (i < start_position or
                                     self.matches < self.min_matches):
            c = self.doc[i]
            self.window_unigrams[c] -= 1
            if self.window_unigrams[c] < self.idea_unigrams[c]:
                self.matches -= 1

            if i + self.window_size < len(self.doc):
                c = self.doc[i + self.window_size]
                self.window_unigrams[c] += 1
                if self.window_unigrams[c] <= self.idea_unigrams[c]:
                    self.matches += 1
            i += 1
        self.i = i

    def slide(self):
        self.slide_to(self.i + 1)

    def next_match(self):
        if self.i >= len(self.doc):
            return None

        assert self.matches >= self.min_matches
        return self.i
