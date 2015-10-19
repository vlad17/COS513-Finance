""" Clean and merge csv files """

from __future__ import division
from os import listdir
import csv
from datetime import datetime
import tldextract
import sys
import os


NUM_FIELDS = 58


def find_csv_filenames(path_to_dir, suffix=".CSV"):
    """ Find all csv files in a directory """

    filenames = listdir(path_to_dir)
    csv_files = [filename for filename in filenames if filename.endswith(suffix)]
    return sorted(csv_files)


def check_row(row):
    """ Check if row is well-formed """

    # length = len(row)
    num_empty = row.count('')
    if num_empty > 5:
        print 'stripped row'
        return False
    return True


def hash_string(s):
    """ hash strings """

    if s == '':
        return 0
    else:
        return hash(s)


def clean_row(row):
    """ Clean a row to [2, 27, 30, 31, 7, 17, 26, 32, 33, 34, 35, 37, 44, 58] """

    row.extend([''] * (NUM_FIELDS - len(row)))

    new_row = []

    sql_date = row[1]
    event_code = row[26]
    quad_class = row[29]
    goldstein = row[30]

    actor_1_name = hash_string(row[6])
    actor_2_name = hash_string(row[16])

    root_event = row[25]

    num_mentions, num_sources, num_articles = row[31], row[32], row[33]
    tone = row[34]
    actor_1_geo_full_name = hash_string(row[36])
    actor_2_geo_full_name = hash_string(row[43])

    url = row[57]

    # format date
    unix_epoch = datetime(1970, 1, 1)
    parsed_sql_date = datetime(int(sql_date[:4]), int(sql_date[4:6]), int(sql_date[6:8]))
    days_since_epoch = (parsed_sql_date - unix_epoch).days
    print days_since_epoch

    # format url
    domain = hash_string(tldextract.extract(url).domain)

    # add other fields
    new_row.extend([days_since_epoch, event_code, quad_class, goldstein, actor_1_name, actor_2_name, root_event, num_mentions, num_sources, num_articles, tone, actor_1_geo_full_name, actor_2_geo_full_name, domain])


    return new_row


def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    output_file = open(output_dir, 'w+')
    csv_writer = csv.writer(os.getcwd() + '/' + output_file, delimiter='\t')

    for csv_filename in find_csv_filenames(input_dir):
        print csv_filename
        with open(input_dir + '/' + csv_filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            for row in reader:
                cleaned_row = clean_row(row)

                if check_row(cleaned_row):
                    csv_writer.writerow(cleaned_row)

    output_file.close()




if __name__ == "__main__":
    main()

















