import numpy
from scipy.misc import logsumexp

def forward_evaluate_log(x, y, pfst_params):
    alpha = numpy.zeros((len(x)+1, len(y)+1), dtype=numpy.float)
    alpha[0,0] = 0.0
    for t in range(len(x)+1):
        for v in range(len(y)+1):

            vals = []
            add = 0.0
            deletion = 0.0
            diag = 0.0
            #if v > 0 or t > 0:
            #    pass
            if v > 0: # addition
                add += numpy.log(pfst_params.trans(None, y[v-1])) + alpha[t,v-1]
                vals.append(add)
            if t > 0:
                deletion += numpy.log(pfst_params.trans(x[t-1], None)) + alpha[t-1,v]
                vals.append(deletion)
            if v > 0 and t > 0:
                diag += numpy.log(pfst_params.trans(x[t-1], y[v-1])) + alpha[t-1,v-1]
                vals.append(diag)

            if len(vals) > 0:
                alpha[t,v] = logsumexp(vals)
    
    alpha[t,v] += numpy.log(0.9)
    return alpha
    

def forward_evaluate(x, y, pfst_params):
    alpha = numpy.zeros((len(x)+1, len(y)+1), dtype=numpy.float)
    alpha[0,0] = 1.
    for t in range(len(x)+1):
        for v in range(len(y)+1):
            if v > 0 or t > 0:
                alpha[t,v] = 0
            if v > 0: # addition
                alpha[t,v] += pfst_params.trans(None, y[v-1]) * alpha[t,v-1] 
            if t > 0:
                alpha[t,v] += pfst_params.trans(x[t-1], None) * alpha[t-1,v]
            if v > 0 and t > 0:
                alpha[t,v] += pfst_params.trans(x[t-1], y[v-1]) * alpha[t-1,v-1]

    
    alpha[t,v] *= .9
    return alpha

def get_all_partitions(wordids):
    start_end_tuples = []
    for end in range(len(wordids)+1):
        for start in range(end):
            start_end_tuples.append((start, end))
    return start_end_tuples

class PartitionIdeaScore:
    def __init__(self, partition, idea_idx, score):
        self.partition = partition
        self.idea_idx = idea_idx
        self.score = score

def get_partition_assignments(part_idea_dict, wordids):
    max_part_dict = {}
    #print part_idea_dict
    for part, idea_dict in part_idea_dict.iteritems():
        max_part_dict[part] = max(idea_dict.items(), key=lambda x:x[1])

    max_to_node = {}
    
    edges = max_part_dict.keys()
    edges.sort(key=lambda x:x[1])
    for terminal_node in range(1, len(wordids)+1):
        max_to_node[terminal_node] = PartitionIdeaScore((0, terminal_node), \
                                                        max_part_dict[(0,terminal_node)][0], \
                                                        max_part_dict[(0,terminal_node)][1])

    for terminal_node in range(1, len(wordids)+1):
        # get the max path to this node
        for start_node in range(1, terminal_node):
            
            path_weight = max_to_node[start_node].score + max_part_dict[(start_node, terminal_node)][1]
            if path_weight > max_to_node[terminal_node].score:
                idea_idx = max_part_dict[(start_node, terminal_node)][0]
                # print max_to_node[terminal_node], path_weight
                max_to_node[terminal_node] = PartitionIdeaScore((start_node, terminal_node), \
                                                                idea_idx, \
                                                                path_weight)

    curr_idx = len(wordids)
    
    partition_idea_score = []
    while curr_idx != 0:
        partition_idea_score.append(max_to_node[curr_idx])
        curr_idx = max_to_node[curr_idx].partition[0]

    partition_idea_score.reverse()
    partitions_assignments = [(p.partition, p.idea_idx) for p in partition_idea_score]
    scores = [p.score for p in partition_idea_score]

    return partitions_assignments, scores
                

def get_part_idea_dict(wordids, ptr_params, pfst_params):
    part_idea_dict = {}
    dp_table_dict = {}
    for s in range(len(wordids)):
        for idea_idx, idea in ptr_params.ideas.items():
            dp_table_dict[idea_idx] = \
                forward_evaluate_log(idea, wordids[s:], pfst_params)

        for e in range(s,len(wordids)+1):
            print s, e, len(wordids)
            part_idea_dict[(s,e)] = {}
            idea_dict = part_idea_dict[(s,e)]
            idea_dict[None] = metrics.get_unigram_logprob(ptr_data, wordids[s:e])
            for idea_idx, idea in ptr_params.ideas.items()[:2]:
                dp_table = dp_table_dict[idea_idx]
                if e - s > 4:
                    if e == len(wordids) + 1:
                        idea_dict[idea_idx] = dp_table[len(idea),e-s]
                    else:
                        idea_dict[idea_idx] = dp_table[len(idea),e-s] + end_log

    return part_idea_dict
    


class AlignScorer:
    def __init__(self, action_prob=None, wordid_count=None):
        
        self.wordid_count = wordid_count

        if wordid_count == None:
            pass
        else:
            self.num_tokens = sum(wordid_count.values())
            self.vocab = len(wordid_count)
            
        self.action_prob = action_prob

    def unigram_score(self, wordid):
        ct = self.wordid_count.get(wordid, 0)
        return numpy.log(ct+1) - numpy.log(self.num_tokens+self.vocab)

    def score_linear(self, op, wordid):
        if op == 'MATCH':
            score = 1.0
        elif op == 'SUB':
            score = -1.0
        elif op == 'DEL':
            score = -1.0
        else:
            assert op == 'ADD'
            score = -1.0
        return score
    
    def score_logprob(self, op, wordid):
        score = 0.0
        if op == 'MATCH':
            score += numpy.log(self.action_prob["MATCH"])
        elif op == 'SUB':
            score += numpy.log(self.action_prob["SUB"]) + \
                               self.unigram_score(wordid)
        elif op == 'DEL':
            score += numpy.log(self.action_prob["SUB"]) + \
                               self.unigram_score(-1)
        else:
            assert op == 'ADD'
            score += numpy.log(self.action_prob["ADD"]) + \
                               self.unigram_score(wordid)
        return score                    
        
class AlignedPassages:

    def __init__(self, passage1, passage2, align_scorer=None):
        self.reference_passage = passage1
        self.query_passage = passage2
        self.align_scorer = align_scorer

    def global_align(self):
        """
        Global alignment with specified scoring function
        """
        passage1 = self.reference_passage
        passage2 = self.query_passage
        
        if self.align_scorer == None:
            self.align_scorer = AlignScorer()
            score = self.align_scorer.score_linear
        else:
            score = self.align_scorer.score_logprob
            
        height = len( passage1 ) + 1
        width = len( passage2 ) + 1

        A = numpy.zeros((height, width), dtype=numpy.float)
        sum_A = numpy.zeros((height, width), dtype=numpy.float)

        backpointers = [ [ (None,None) for c in range( width ) ] for r in range( height ) ]
        operations_matrix = [ [ None for c in range( width ) ] for r in range( height ) ]

        # opening a horizontal gap
        for r in range( height ):
            if r == 0:
                A[r][0] == 0.0
                sum_A[r][0] = 0.0
            else:
                A[r][0] = A[r-1][0] + score("DEL", passage1[r-1])
                sum_A[r][0] = sum_A[r-1][0] + score("DEL", passage1[r-1])
            if r != 0:
                backpointers[r][0] = (-1,0)
                operations_matrix[r][0] = "DEL"

        # opening a vertical gap
        for c in range( width ):
            if c == 0:
                pass
            else:
                A[0][c] = A[0][c-1] + score("ADD", passage2[c-1])
                sum_A[0][c] = sum_A[0][c-1] + score("ADD", passage2[c-1])
            if c != 0:
                backpointers[0][c] = (0,-1)
                operations_matrix[0][c] = "ADD"

        #print passage1, passage2
        #print height*width, height, width
        for r in range( 1, height ):
            for c in range( 1, width ):
                if passage1[r-1] == passage2[c-1]:
                    #arrow = (-1,-1)
                    diag_score = A[r-1,c-1] + score("MATCH",passage1[r-1])
                    match = True
                else:
                    diag_score = A[r-1,c-1] + score("SUB",passage2[c-1])
                    match = False

                # insertion
                horiz_score = A[r,c-1] + score("ADD",passage2[c-1])

                # deletion
                vert_score = A[r-1,c] + score("DEL",passage1[r-1])


                arrows_scores_operations = [ ( (-1,-1), diag_score, "DIAG" ), ( (0,-1), horiz_score, "ADD" ), \
                                 ( (-1,0), vert_score, "DEL" ) ]

                    
                max_arrow, max_score, operation = max( arrows_scores_operations, key=lambda x:x[1] )



                if operation == "DIAG":
                    if match == True:
                        operation = "MATCH"
                    else:
                        operation = "SUB"

                total_score = logsumexp([i[1] for i in arrows_scores_operations])
                #print total_score                  
                A[r,c] = max_score
                sum_A[r,c] = total_score

                backpointers[r][c] = max_arrow
                operations_matrix[r][c] = operation

        # follow back-pointers to None
        final_score = A[-1,-1]

        self.final_score = A[-1,-1]
        self.total_final_score = sum_A[-1,-1]
        
        #print len(self.query_passage)
        #print numpy.exp(self.final_score)
        #print numpy.exp(self.total_final_score)
        #print

        reverse_cell_list = []

        current_row = len( passage1 )  # python last row
        current_col = len( passage2 )
        curr_pointer = backpointers[current_row][current_col]

        #print

        reverse_cell_list.append( ( current_row, current_col ) )
        while curr_pointer != (None,None):
            current_row = current_row + curr_pointer[0]
            current_col = current_col + curr_pointer[1]
            reverse_cell_list.append( ( current_row, current_col ) )
            #print curr_pointer, reverse_cell_list
            curr_pointer = backpointers[current_row][current_col]

        reverse_cell_list.reverse()


        # reference
        passage1_indices = [ i[0] for i in reverse_cell_list ]
        passage1_aligned_list = self.get_aligned_list( passage1_indices, passage1 )

        passage2_indices = [ i[1] for i in reverse_cell_list ]
        passage2_aligned_list = self.get_aligned_list( passage2_indices, passage2 )

        self.aligned_reference_passage = passage1_aligned_list
        self.aligned_query_passage = passage2_aligned_list

        self.cell_list = reverse_cell_list

        self.A_matrix = A
        self.sum_A_matrix = sum_A

        self.backpointers = backpointers
        self.operations_matrix = operations_matrix

        self.get_alignment_operations()

    def get_aligned_list( self, passage_indices, passage ):
        passage_aligned_list = []
        for i in range( len( passage_indices ) ):
            if i == 0:
                assert passage_indices[i] == 0
            else:
                prev_idx = passage_indices[i-1]
                curr_idx = passage_indices[i]
                if curr_idx == prev_idx + 1:
                    passage_aligned_list.append( passage[curr_idx-1] )
                elif curr_idx == prev_idx:
                    passage_aligned_list.append( "-" )
                else:
                    raise Error

        return passage_aligned_list


    def get_alignment_operations( self ):
        self.alignment_operations = []
        assert self.cell_list[0] == (0,0)
        assert len( self.cell_list[1:] ) == len( self.aligned_reference_passage )
        assert len( self.cell_list[1:] ) == len( self.aligned_query_passage )

        for row, col in self.cell_list[1:]:
            self.alignment_operations.append( self.operations_matrix[row][col] )



def lcs(a,b):
    """ Computes and returns the length of the longest common subsequence """

    dp = [0] * (len(b)+1)

    for i in range(len(a)):
        ndp = [0] * (len(b)+1)
        for j in range(1, len(dp)):
            ndp[j] = max(dp[j], ndp[j-1])
            if a[i] == b[j-1]:
                ndp[j] = max(ndp[j], dp[j-1]+1)
        dp = ndp

    return dp[-1]


def main():

    x_ap = AlignedPassages( ["hello"], ["hello", "goodbye"] )
    x_ap = AlignedPassages( [ "BOB", "DOLE", "WON", "THE", "ELECTION", "IN", "LANDSLIDE" ], [ "DOLE", "WON", "ELECTION", "BY", "A", "LANDSLIDE" ] )
    #print x_ap.reference_passage
    #print x_ap.query_passage
    x_ap.global_align()
    #x_ap.A_matrix
    print x_ap.aligned_reference_passage
    print x_ap.aligned_query_passage
    #x_ap.get_alignment_operations()

    #for r in x_ap.alignment_operations:
    #    print r

    print x_ap.alignment_operations


if __name__ == '__main__':
    main()
