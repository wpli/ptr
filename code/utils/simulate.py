import PTRExperiment
import synthetic
import align
import string
import random
import sys
import itertools

def get_all_ideas():
    s1 = "Bob Dole won the election handily in 1996."
    s2 = "There are some problems that not even $10 trillion can solve."
    s3 = "The Supreme Court ruled on Monday against three death row inmates who had sought to bar the use of an execution drug they said risked causing excruciating pain."
    s4 = "China is spending hundreds of billions of dollars annually in an effort to become a leader in biomedical research, building scores of laboratories and training thousands of scientists."
    s5 = "After tunneling out of maximum-security cells, Richard W. Matt and David Sweat waited for the Jeep, driven by a cooperating prison employee, that would take them to Mexico."
    s6 = "Suppose your teacher told you that an assignment was due by a certain day."
    s7 = "This summer, during your sweaty subway ride to work or moments waiting for that ride to the lake, we invite you to hang out with us."
    all_ideas = [ i.split() for i in ( s1, s2, s3, s4, s5, s6, s7 ) ]
    return all_ideas

# generate synthetic data
def sim( all_ideas, max_partitions=50, num_docs=500, debug=False ):
    # generate vocabulary
    num_words = 1000
    vocab_words_set = set()

    num_ideas = len( all_ideas )

    idea_bank = all_ideas[:num_ideas]

    for idea in idea_bank:
        for word in idea:
            vocab_words_set.add( word )

    consonants = list( set(string.ascii_lowercase) - set('aeiou') )
    vowels = list( set( 'aeiou' ) )

    while 1:
        rand_cons = [ random.choice(consonants) for j in range(3) ]
        rand_cons.insert( 2, random.choice( vowels ) )
        rand_cons.insert( 1, random.choice( vowels ) )
        rand_word = "".join( rand_cons )

        if rand_word not in vocab_words_set:
            vocab_words_set.add( rand_word )

        if len( vocab_words_set ) == num_words:
            break

    vocab_words = list( vocab_words_set )
    vocab_words.sort()

    # store vocabulary and number of tokens
    synthetic_data = PTRExperiment.PTRData()
    synthetic_data.vocab = vocab_words
    synthetic_data.word_wordid = dict([reversed(i) for i in enumerate(vocab_words)])
    synthetic_data.num_tokens = num_words

    occurrence_of_ideas = [ .05 for i in idea_bank ]
    synthetic_data_initial_params = PTRExperiment.PTRParameters()
    for idx, idea in enumerate( idea_bank ):
        idea_wordids = tuple( [ synthetic_data.word_wordid[i] for i in idea ] )
        synthetic_data_initial_params.ideas.add_idea( idea_wordids, idx, occurrence_of_ideas[idx] )
        synthetic_data_initial_params.max_partition_length = 40

    # generate the data
    #sys.stderr.write( "Generating data..." )
    synthetic_doc_dict_list = synthetic.generate_synthetic_data( synthetic_data, synthetic_data_initial_params, \
                                                                 max_partitions=max_partitions, num_docs=num_docs, debug=debug )
    #sys.stderr.write( "done.\n" )


    # add synthetic documents to dataset
    sys.stderr.write( "Generating PTRData..." )
    for idx, doc_dict in enumerate( synthetic_doc_dict_list ):
        full_doc = list( itertools.chain( *doc_dict['partition_contents'] ) )
        #for contents in doc_dict['partition_contents']:
        #    full_doc += contents

        synthetic_data.add_doc( idx, full_doc )
    sys.stderr.write( "done.\n" )



    for idx, doc_dict in enumerate( synthetic_doc_dict_list ):
        #if idx % 10 == 0:
        #    sys.stderr.write( "%s " % ( idx ) )
        docid = idx

        partition_assignment = {}
        current_start_idx = 0
        #print synthetic_data.docid_wordids[docid]
        for contents, assignment in zip( doc_dict['partition_contents'], doc_dict['partition_assignments'] ):
            current_end_idx = current_start_idx + len( contents )
            tup = ( current_start_idx, current_end_idx )
            assert assignment < len( synthetic_data_initial_params.ideas ) or assignment == None
            partition_assignment[tup] = assignment

            current_start_idx = current_end_idx

        synthetic_data_initial_params.docid_partitions[docid] = partition_assignment

    return synthetic_data, synthetic_data_initial_params



