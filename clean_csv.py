""" Clean and merge csv files """

from __future__ import division
from os import listdir
from os import getcwd
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


def clean_row(row):
    """ Clean a row to [2, 27, 30, 31, 7, 17, 26, 32, 33, 34, 35, 37, 44, 58] """

    if len(row) < NUM_FIELDS:
        row.extend('')

    new_row = []


    # format date
    unix_epoch = datetime(1970, 1, 1)
    sql_date = row[1]
    parsed_sql_date = datetime(int(sql_date[:4]), int(sql_date[4:6]), int(sql_date[6:8]))
    days_since_epoch = (parsed_sql_date - unix_epoch).days
    print days_since_epoch

    new_row.extend([days_since_epoch])

    # add other fields
    new_row.extend([row[26], row[29], row[30], row[6], row[16], row[25], row[31], row[32], row[33], row[34], row[36], row[43]])

    # format url
    domain = tldextract.extract(row[57]).domain
    new_row.extend([domain])

    return new_row


def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    output_file = open(output_dir, 'w+')
    csv_writer = csv.writer(output_file, delimiter='\t')

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

















