## Input file: ../data/train.csv
## Output file: ../data/train_with_augmented_ticket_features.csv
##
## Extracts features from ticket numbers:
## 	- actual number of the ticket;
##	- logical features (0/1) based on the string prefix of the ticket. Apparently, the prefix contains information about the ticket seller (or selling place). 
##
## Example: 
## ticket number "C.A./SOTON 34068"
##
## - actual number is 34068
## - string prefix is "casoton"
##
## New features extracted from it:
##	ticket_number	 	34068
##	ticket_from_casoton	1
##	ticket_from_a		0
##	ticket_from_pp		0
##	...			0
## 


def extract_digits(s):
    ''' Extracts actual number of the ticket, skipping the optional string prefix.
        Examples:
        >>> extract_digits("C.A./SOTON 34068")
        34068
        >>> extract_digits("34568")
        36568
    '''
    ends_with = s.split()[-1]
    if ends_with.isdigit():
        return int(ends_with)
    else:
        return 0


def extract_text(s):
    ''' Removes digits, punctuation and trailing spaces from a string. Makes the result  lowercase.
        Examples: 
        >>> extract_text('STON/O2. 3101282')
        'stono'
        >>> extract_text('S.C./PARIS 2079')
        'scparis'

        This function produced the following features: 
        set(['scah', 'soc', 'ca', 'sca', 'sop', 'as', 'ppp', 'scparis', 'line', 'scahbasle', 'pp', 'scow', 'pc', 'wep', 'sotonoq', 'sopp', 'wc', 'fa', 'sotono', 'fcc', 'casoton', 'swpp', 'a', 'c', 'stono', 'sp', 'fc', 'sc'])
         ===>
         1. Additional cleaning may be needed, e.g. "sotono" and "stono" are the same place  (Southampton)
         2. SC/AH and SC/PARIS: both contain "SC"; should we do anything with it? Same case with C.A./SOTON 
    '''
    return ''.join(i for i in s if not (i.isdigit() or i in  '., /?!:;')).strip().lower()



import csv

with open('../data/train.csv','rb') as input_file:
    reader = csv.DictReader(input_file)

    ticket_features = set('') # empty set
    for row in reader:
        ticket_features.add(extract_text(row['ticket']))

    ticket_features -= set(['']) # remove the empty string
    
    input_file.seek(0) # rewind
    reader.next()      # skip the header line

    with open('../data/train_with_augmented_ticket_features.csv', 'wb') as output_file:
        writer = csv.DictWriter(output_file, 
                  fieldnames = reader.fieldnames 
                               + ['ticket_number'] 
                               + map(lambda s: 'ticket_from_'+s, ticket_features)) 
                                # ^ add 'ticket_from_' to feature names
        writer.writeheader()

        for row in reader:
            row_new = row
            ticket = row['ticket']

            row_new['ticket_number'] = extract_digits(ticket)
            ticket_text = extract_text(ticket)

            for feature in ticket_features:
                row_new['ticket_from_'+feature] = 1 if ticket_text == feature else 0

            writer.writerow(row_new)

