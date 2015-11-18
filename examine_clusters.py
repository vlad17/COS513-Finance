
"""
Usage: python examine_clusters.py raw_data_dir model_file 

Examines cluster model for a given K

Example: python examine_clusters.py /n/fs/gcf/raw-data /n/fs/scratch/dchouren/models/1000.model
"""

from glob import glob
import pickle
import sys
import os


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


    print('Preprocessing')
    files = glob(preprocessed_dir + '/*')
    for f in files:
        os.remove(f)
    preprocess_files = glob(preprocessed_dir + '/*')

    for inf in raw_files:
        # print inf
        # print pre_process_dir + '/' + inf.split('/')[-1]
        os.system('python preprocessing.py {} {}'.format(inf, preprocessed_dir + '/' + inf.split('/')[-1]))

    print('Expanding')
    files = glob(expanded_dir + '/*')
    for f in files:
        os.remove(f)

    for inf in preprocess_files:
        os.system('python expand.py {} {}'.format(inf, expanded_dir + '/' + inf.split('/')[-1]))

    expanded_files = glob(expanded_dir + '/*')


    print('Examining')
    for raw_file, expanded_file in zip(raw_files, expanded_files):
        with open(raw_file, 'r') as raw_in, open(expanded_file, 'r') as expanded_in:
            for raw_line, expanded_line in raw_in, expanded_in:
                print(raw_line)
                print(model.predict(expanded_line))

    return 0

if __name__ == "__main__":
    sys.exit(main())



