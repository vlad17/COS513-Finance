
"""
Usage: python examine_clusters.py raw_data_dir preprocessed_dir expanded_dir model_file 

Examines cluster model assignments by preprocessing, expanding, and then clustering some raw data according to a loaded model

Example: python examine_clusters.py /n/fs/scratch/dchouren/examine_data/raw /n/fs/scratch/dchouren/examine_data/preprocessed/ /n/fs/scratch/dchouren/examine_data/expanded/ /n/fs/gcf/dchouren-repo/COS513-Finance/models/1000.model
"""

from glob import glob
import pickle
import sys
import os
import ipdb


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    raw_dir = sys.argv[1]
    preprocessed_dir  = sys.argv[2]
    expanded_dir = sys.argv[3]
    pickled_model = sys.argv[4]

    print()
    print("Loading model")
    model = None
    with open(pickled_model, 'rb') as model_file:
        model = pickle.load(model_file)

    raw_files = glob(raw_dir + '/*')

    print()
    print('Preprocessing')
    files = glob(preprocessed_dir + '/*')
    for f in files:
        os.remove(f)

    for inf in raw_files:
        os.system('python preprocessing.py {} {}'.format(inf, preprocessed_dir + inf.split('/')[-1]))
    
    preprocess_files = glob(preprocessed_dir + '/*')

    print()
    print('Expanding')
    files = glob(expanded_dir + '/*')
    for f in files:
        os.remove(f)

    for inf in preprocess_files:
        os.system('python expand.py {} {}'.format(inf, expanded_dir + '/' + inf.split('/')[-1]))

    expanded_files = glob(expanded_dir + '/*')

    print()
    print('Examining')
    clusters = {}

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    for raw_file, expanded_file in zip(raw_files, expanded_files):
        with open(raw_file, 'r') as raw_in, open(expanded_file, 'r') as expanded_in:
            for raw_line, expanded_line in zip(raw_in, expanded_in):
                cluster = int(model.predict(expanded_line.split('\t')[7:])[0])
                try:
                    clusters[cluster].append(raw_line.split('\t')[-1])
                except KeyError:
                    clusters[cluster] = [raw_line.split('\t')[-1]]

    sys.stdout = save_stdout

    ipdb.set_trace()

    with open('/n/fs/gcf/dchouren-repo/COS513-Finance/models/clusters.pickle', 'wb') as handle:
        pickle.dump(clusters, handle)

    return 0

if __name__ == "__main__":
    sys.exit(main())



