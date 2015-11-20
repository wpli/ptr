import string
def strip_punctuation( input_str ):
    out = input_str.translate(string.maketrans("",""), string.punctuation)
    out = out.lower()
    return out

def jaccard( set1, set2 ):
    return float( len( set.intersection( set1, set2 ) ) ) / len( set.union( set1, set2 ) )


def unit_test():
    unit_test_jaccard()

def unit_test_jaccard():
    sample_sentence = "the quick brown fox jumps over lazy dog"
    words = sample_sentence.split()

    set1 = set( words[:5] )
    set2 = set( words[3:] )
    print set1
    print set2
    print jaccard( set1, set2 )
    

if __name__ == '__main__':
    unit_test()
