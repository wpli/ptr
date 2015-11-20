import csv

"""
Input: A CSV file (in Excel dialect) with headers
Returns the schema as a list
"""
def get_schema( filename, dialect='excel' ):
    with open( filename ) as f:
        csvreader = csv.reader( f, dialect )
    
        header = next( csvreader )
        header = [ i.replace( ' ', '_' ) for i in header ]

    return header
