"""
Usage: python random_sample.py infile_dir outfile N R K

N = total number of events to sample
R = total number of files to sample
K = number of clusters

Serializes (via pickle) the K means model based on the input to outfile.
"""

import glob
import sys
import random
import re
import ipdb
import os




TOPIC_COLUMNS = 1138


def main():

    if len(sys.argv) != 6:
        print(__doc__)
        return 1

    begin_year = 2006
    end_year = 2011

    infiles = set(glob.glob(sys.argv[1] + '/*'))
    outfile = sys.argv[2]
    N = int(sys.argv[3])
    R = int(sys.argv[4])
    K = int(sys.argv[5])

    num_file_matches = 0
    file_matches = {}

    print("Sampling {} files for {} total events".format(R, N))
    total_num_events = 0
    while num_file_matches < R:   # R-1 because we have one final sample to account for int division rounding down
        random_file = random.choice(list(infiles))

        infiles.remove(random_file)


        if re.match('^(2006|2007|2008|2009|2010|2011)', random_file.split('/')[-1]):
            num_events = 0
            with open(random_file, 'r') as inf:
                num_events = sum(1 for _ in inf)
            total_num_events += num_events
            file_matches[random_file] = num_events

            num_file_matches += 1

    total_selected = 0

    first_file = None
    os.remove(outfile)
    for filename, num_events in file_matches.items():
        if first_file == None:
            first_file = filename
        num_to_sample = int(num_events / float(total_num_events) * N)
        total_selected += num_to_sample

        os.system('gshuf -n {} {} >> {}'.format(num_to_sample, first_file, outfile))

    # since we used integer division to calculate the number of lines to sample from each file
    # we need to oversample for the last file
    remaining_to_sample = N - total_selected

    os.system('gshuf -n {} {} >> {}'.format(remaining_to_sample, first_file, outfile))


    return 0

if __name__ == "__main__":
    sys.exit(main())













