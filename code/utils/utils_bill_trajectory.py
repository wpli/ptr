import os, sys
import csv
import datetime
import dateutil
import cPickle
sys.path.append( '../utils' )
import utils_text
import utils_search
import collections

#data_path = '../../data/unshared-task-poliinformatics-2014-v1.0/CongressionalBills'
#bills = [ i for i in os.listdir( data_path ) if ".txt" in i ]
sys.path.append( '../utils' )
csv.field_size_limit(sys.maxsize)

#congress_files = [ '110thALL.txt', '111thALL.txt' ]

def get_data_list_of_dicts( data_path, congress_files ):
    """Load the bill text entries as dictionaries."""
    data_list_of_dicts = []
    for congress in congress_files:    
        full_path = os.path.join( data_path, congress )
        with open( full_path, 'rb' ) as f:
            csvreader = csv.reader( f, delimiter='\t', quoting=csv.QUOTE_ALL )
            for idx, row in enumerate(  csvreader ):
                if idx == 0:
                    header = row
                else:    
                    data_dict = dict( zip( header, row ) )
                    assert len( data_dict ) == len( header )
                    data_list_of_dicts.append( data_dict )
    return data_list_of_dicts

def get_data_list_sets( data_list_of_dicts ):
    data_list_sets = []
    for idx, i in enumerate( data_list_of_dicts ):
        if idx % 10000 == 0:
            sys.stderr.write( "%s " % idx )
        x = i['text'].lower()
        text = utils_text.strip_punctuation( x )
        word_set = set( text.split() )
        data_list_sets.append( word_set )
    return data_list_sets


def get_word_inverted_index( data_list_sets ):
    word_inverted_index = collections.defaultdict( list )
    for idx, word_set in enumerate( data_list_sets ):
        if idx % 10000 == 0:
            sys.stderr.write( "%s " % idx )

        for word in list( word_set ):
            word_inverted_index[word].append( idx )
    return word_inverted_index


def get_target_top_matches( target_indices, data_list_sets, word_inverted_index ):
    target_top_matches = []

    for idx in sorted( list( target_indices ) ):
        word_set = data_list_sets[idx]
        sys.stderr.write( "%s " % idx )        
        matches = utils_search.get_ranked_matches( word_set, word_inverted_index, top_k=100 ) 
        target_top_matches.append( ( idx, matches ) )

    return target_top_matches

def get_target_top_matches_jaccard( target_top_matches, data_list_sets ):
    target_top_matches_jaccard = []

    for query_idx, matches in target_top_matches:
        query_word_set = data_list_sets[query_idx]
        jaccard_scores = []
        for match_idx, num_matches in matches:
            match_word_set = data_list_sets[match_idx]
            jacc = utils_text.jaccard( query_word_set, match_word_set )
            jaccard_scores.append( jacc )

        match_jaccard = zip( [ i[0] for i in matches ], jaccard_scores )
        match_jaccard.sort( key=lambda x:x[1], reverse=True )

        target_top_matches_jaccard.append( ( query_idx, match_jaccard ) )

    return target_top_matches_jaccard


def get_section_intro_dates( target_bill_date, datetimes_sections ):
    section_numbers = set( [ i[1] for i in datetimes_sections ] )
    num_sections = len( section_numbers )
    dates_by_section_dict = collections.defaultdict( list )
    for dt, section in datetimes_sections:
        dates_by_section_dict[section].append( dt.date() )
        
    section_intro_dates = {}
    for section, dates in dates_by_section_dict.items():
        section_intro_dates[section] = min( dates )
        
    assert len( section_intro_dates ) == num_sections 
    
    return section_intro_dates

def get_week_threshold_metric( section_intro_dates, target_bill_date, week_threshold=12 ):
    num_weeks_before_threshold = 0
    for section, intro_date in section_intro_dates.items():
        timedelta = target_bill_date - intro_date
 
        weeks = timedelta.days / 7.0
        if weeks > week_threshold:
            num_weeks_before_threshold += 1
            
    week_threshold_metric = float( num_weeks_before_threshold ) / len( section_intro_dates )
    
    return week_threshold_metric

def get_average_gestation_metric( section_intro_dates, target_bill_date ):
    section_weeks_before_dict = {}
    for section, intro_date in section_intro_dates.items():
        timedelta = target_bill_date - intro_date
        weeks = timedelta.days / 7.0
        section_weeks_before_dict[section] = weeks
        
    average_gestation_metric = sum( section_weeks_before_dict.values() ) / len( section_weeks_before_dict )
    return average_gestation_metric

def is_match( match_entry, target_bill_date, query_entry, jaccard, include_future=False ):
    match = True
    match_date_text = match_entry['IssuedOn']

    match_text = match_entry['text']
    query_text = query_entry['text']
    num_words_match = len( match_text.split() )
    num_words_query = len( query_text.split() )
    if match_date_text == '0000-00-00':
        match = False
    else:
        match_datetime = datetime.datetime.strptime( match_date_text, "%Y-%m-%d" )
        match_date = match_datetime.date()
        
        if match_date > target_bill_date and include_future == False:
            match = False
        elif "SEVERABILITY" in match_text:
            match = False
        elif jaccard == 1.0:
            match = True
        elif num_words_query < 500:
            if jaccard < 0.88:
                match = False
        elif jaccard < 0.75:
            match = False

    
    return match
