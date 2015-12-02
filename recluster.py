from glob import glob
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import itertools
import fileinput
import pickle
import sys
import glob
import os
import re
import ipdb

def main():

    infiles = set(glob.glob(sys.argv[1] + '/*'))
    outfile = sys.argv[2]

    try:
        os.remove(outfile)
    except OSError:
        pass


    # filtered_in = []

    # for inf in infiles:
    #     if re.match('^(201007)', inf.split('/')[-1]):

    #         filtered_in.append(inf)

    # ipdb.set_trace()

    # print("Reading in", len(filtered_in), "files")
    # fullarr = np.loadtxt(fileinput.input(filtered_in), delimiter = '\t')[:,7:]


    K = 100
    print("Learning MiniBatchKMeans with K =", K)

    # km = MiniBatchKMeans(n_clusters = K, verbose = True)
    # km.fit(fullarr)
    km = pickle.load(open('/n/fs/gcf/dchouren-repo/COS513-Finance/new100.model2', 'rb'))

    # with open(outfile, 'wb') as out_model:
    #     pickle.dump(km, out_model)


    print('Examining')
    clusters = {}

    expanded_dir = '/n/fs/scratch/dchouren/examine_data/expanded'
    expanded_files = glob.glob(expanded_dir + '/*')
    raw_dir = '/n/fs/scratch/dchouren/examine_data/raw'
    raw_files = glob.glob(raw_dir + '/*')

    num_spread = 0
    clusters_in = []

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    for raw_file, expanded_file in zip(raw_files, expanded_files):
        with open(raw_file, 'r') as raw_in, open(expanded_file, 'r') as expanded_in:
            for raw_line, expanded_line in zip(raw_in, expanded_in):
                cluster = int(km.predict(expanded_line.split('\t')[7:])[0])

                url = raw_line.split('\t')[-1]
                if 'lakhvi' in url.lower():
                    clusters_in.append(cluster)
                    # print(cluster)
                try:
                    clusters[cluster].append(url)
                except KeyError:
                    clusters[cluster] = [url]

    sys.stdout = save_stdout

    print(clusters_in)
    print(len(clusters_in))

    ipdb.set_trace()
    ipdb.set_trace()

if __name__ == "__main__":
    sys.exit(main())
