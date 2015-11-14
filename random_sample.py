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
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle



begin_year = 2006
end_year = 2011

infiles = set(glob.glob(sys.argv[1] + '/*'))
outfile = sys.argv[2]
N = int(sys.argv[3])
R = int(sys.argv[4])
K = int(sys.argv[5])


TOPIC_COLUMNS = 1138


def main():
    num_file_matches = 0
    file_matches = {}

    total_num_events = 0
    while num_file_matches < R:
        random_file = random.choice(list(infiles))

        infiles.remove(random_file)
        # print(random_file)

        if re.match('^(2006|2007|2008|2009|2010|2011)', random_file.split('/')[-1]):
            num_events = 0
            with open(random_file, 'r') as inf:
                num_events = sum(1 for _ in inf)
            total_num_events += num_events
            file_matches[random_file] = num_events

            num_file_matches += 1

    out_events = []
    total_selected = 0
    for filename, num_events in file_matches.items():
        num_to_sample = int(num_events / float(total_num_events) * N)
        total_selected += num_to_sample
        with open(filename, 'r') as inf:
            lines = inf.readlines()
            random_lines = random.sample(lines, num_to_sample)
            out_events.extend(random_lines)

    remaining_to_sample = N - total_selected
    filename = file_matches.popitem()[0]
    with open(filename, 'r') as inf:
        lines = inf.readlines()
        random_lines = random.sample(lines, remaining_to_sample)
        out_events.extend(random_lines)


    fullarr = np.array(out_events)

    with open(outfile, 'w') as outf:
        outf.writelines(out_events)

    del out_events


    print("Learning MiniBatchKMeans with K =", K)

    km = MiniBatchKMeans(n_clusters = K, verbose = True) # TODO max_iter
    km.fit(fullarr)

    print("KMeans trained, saving")

    with open(outfile, 'wb') as out_model:
        pickle.dump(km, out_model)

    print("Score:", km.score(fullarr))

    return 0

if __name__ == "__main__":
    sys.exit(main())













