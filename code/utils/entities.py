
# coding: utf-8

# In[1]:

class IdeaBank( dict ):

    
    def __init__( self ):
        self.active_ideas = set()
        self.idea_counter = 0
        self.inactive_ideas = set()
        self.wordids_idx_dict = {}
        
    def add_idea( self, word_ids ):
        tuple_word_ids = tuple( word_ids )
        self[self.idea_counter] = tuple_word_ids
        self.active_ideas.add( self.idea_counter )
        self.wordids_idx_dict[word_ids] = self.idea_counter
        self.idea_counter += 1

        
    def deactivate_idea_by_index( self, idx ):
        self.active_ideas.remove( idx )
        self.inactive_ideas.add( idx )
        assert len( set.intersection( self.active_ideas, self.inactive_ideas) ) == 0
        assert len( set.union( self.active_ideas, self.inactive_ideas) ) == len( self )
    
    def get_active_idea_set( self ):
        return self.active_ideas
    
        
    
    
    
    

get_ipython().system(u'ipython nbconvert --to python entities.ipynb')


# In[58]:

def unit_test():
    ideas = IdeaBank()
    import cPickle
    with open( "tmp.pickle", 'w' ) as f:
        cPickle.dump( ideas, f )

    with open( "tmp.pickle" ) as f:
        ideas2 = cPickle.load( f )

    print ideas, ideas2
    
def unit_test_make_dict():
    ideas = IdeaBank()
    ideas.add_idea( (1,2,3) )
    ideas.add_idea( (2,3) )
    #ideas.deactivate_idea_by_index( 0 )
    print ideas.active_ideas

#unit_test_make_dict()


# In[37]:

#unit_test()


# In[37]:




# In[ ]:



