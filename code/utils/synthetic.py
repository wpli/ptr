import random
import numpy
import sys


def generate_synthetic_data( ptr_data, ptr_params, max_partitions=50, num_docs=1000, debug=False ):
    # generate documents
    vocab_words = ptr_data.vocab

    max_partition_length = ptr_params.max_partition_length
    idea_bank = ptr_params.ideas

    idea_prob_tuples = ptr_params.ideas.idea_prob.items()
    ideas = [ i[0] for i in idea_prob_tuples ]
    idea_probs = [ i[1] for i in idea_prob_tuples ]

    occurrence_of_ideas = ptr_params.ideas.idea_prob
    action_prob_tuples = ptr_params.action_prob.items()
    num_ideas = len( idea_bank )

    
    actions = [ i[0] for i in action_prob_tuples ]
    action_probs = [ i[1] for i in action_prob_tuples ]
    
    doc_dict_list = []

    for d in range( num_docs ):
        if debug:
            sys.stderr.write( "Generating document %s\n" % d )
        
        doc_dict = {}
        # generate the number of partitions

        num_partitions = random.randint( 1, max_partitions )

        if debug:
            sys.stderr.write( "Number of partitions: %s\n" % num_partitions )
            
            
            
        doc_dict['num_partitions'] = num_partitions

        # with the number of partitions, determine whether part of idea

        partition_assignments = []
        partition_contents = []
        partition_perturbations = []
        
        for p in range( num_partitions ):    
            # choose which idea 
            selected_idea = numpy.random.choice( ideas, p=idea_probs )
            partition_assignments.append( selected_idea )

            # background assignment
            
            if debug:
                sys.stderr.write( "Partition %s, selected idea: %s\n" % ( p, selected_idea ) )

            part_words = []
            if selected_idea == None:
                partition_perturbations.append( None )

                # choose a length
                partition_length = random.randint( 1, max_partition_length )

                # generate words 
                for word in range( partition_length ):
                    word = random.choice( vocab_words )
                    part_words.append( word )
                if debug:
                    sys.stderr.write( "selected length: %s, words: %s\n\n" % ( partition_length, part_words ) )

                    
            else:
                # assignment to the idea
                idea = ptr_data.get_words_from_wordids( idea_bank[selected_idea] )
                
                perturbs = []

                for word in idea:
                    action = numpy.random.choice( actions, p=action_probs )
                    perturbs.append( action )
                    if action == "MATCH":
                        part_words.append( word )
                    elif action == "SUB":
                        substituted_word = random.choice( vocab_words + [ "" ] )
                    elif action == "ADD":
                        part_words.append( word )
                        added_word = random.choice( vocab_words )
                        part_words.append( added_word )
                    else:

                        raise Error

                if debug:
                    sys.stderr.write( "perturbations: %s\n" % ( perturbs ) ) 
                    sys.stderr.write( "words: %s\n\n" % ( part_words, ) )
                    

                    
                partition_perturbations.append( perturbs )
                #print " ".join(part_words)
            partition_contents.append( part_words )

        doc_dict['partition_perturbations'] = partition_perturbations    
        doc_dict['partition_assignments'] = partition_assignments
        doc_dict['partition_contents'] = partition_contents
        assert len( partition_assignments ) == len( partition_contents )
        doc_dict_list.append( doc_dict )
    return doc_dict_list

def get_single_doc():
    synthetic_doc_dict_list = [ { 'partition_assignments': [ 0 ], 
                              'partition_contents': [ idea_bank[0] ],
                              'partition_perturbations': [ "MATCH" for i in idea_bank[0] ],
                             'num_partitions': 1 
                            }    ] 

    return synthetic_doc_dict_list
