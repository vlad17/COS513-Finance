"""
Usage: python random_sample.py infile_dir outfile N

N = total number of events to sample
R = total number of files to sample

Examle: python random_sample.py /n/fs/gcf/raw-data /n/fs/scratch/dchouren/random_20000101.export.CSV 1000000
"""

import glob
import sys
import random
import re
import ipdb
import os


def main():

    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    begin_year = 2006
    end_year = 2011

    infiles = set(glob.glob(sys.argv[1] + '/*'))
    outfile = sys.argv[2]
    N = int(sys.argv[3])

    try:
        os.remove(outfile)
    except OSError:
        pass

    file_matches = {}

    print("Sampling for {} total events".format(N))
    total_num_events = 0

    for inf in infiles:
        if re.match('^(2006|2007|2008|2009|2010|2011)', inf.split('/')[-1]):

            num_events = 0
            with open(inf, 'r') as inf:
                num_events = sum(1 for _ in inf)
            total_num_events += num_events

            file_matches[inf] = num_events


    total_selected = 0

    first_file = None

    print("Shuffing")

    for filename, num_events in file_matches.items():
        if first_file == None:
            first_file = filename
        num_to_sample = int(num_events / float(total_num_events) * N)
        total_selected += num_to_sample

        os.system('shuf -n {} {} >> {}'.format(num_to_sample, filename, outfile))

    # since we used integer division to calculate the number of lines to sample from each file
    # we need to oversample for the last file
    remaining_to_sample = N - total_selected

    os.system('shuf -n {} {} >> {}'.format(remaining_to_sample, first_file, outfile))


    return 0

if __name__ == "__main__":
    sys.exit(main())













