"""
Usage: python summarize.py infile outfile modelfile

Loads via pickle the MiniBatchKMeans model which should be used to identify the
label of the each news item in the infile csv, which should be an "expanded"
format csv, a tab-separated float array.
"""

from sklearn.preprocessing import scale
import csv
import numpy as np
import pickle
import sys
from contextlib import contextmanager
from timeit import default_timer
import time 

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

TOPIC_COLUMNS = 938 # TODO Increase by 200 with word2vec 

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    infile = sys.argv[1]
    outfile = sys.argv[2]
    modelfile = sys.argv[3]

    print('Reading in day file {}... '.format(infile), end = '')
    with elapsed_timer() as elapsed, open(infile, 'r') as i:
        topics, importance = np.hsplit(np.loadtxt(i, delimiter = '\t'), 
                                       [TOPIC_COLUMNS])
    print('{}s'.format(elapsed()))

    print('topics {} importance {}'.format(topics.shape, importance.shape))

    print('Scaling topics matrix... ', end = '')
    with elapsed_timer() as elapsed:
        scale(topics, copy = False)
    print('{}s'.format(elapsed()))

    print('Loading model... ', end = '')
    with elapsed_timer() as elapsed, open(modelfile, 'rb') as i:
        km = pickle.load(i)
    print('{}s'.format(elapsed()))

    print('Predicting data... ', end = '')
    with elapsed_timer() as elapsed:
        predictions = km.predict(topics)
    print('{}s'.format(elapsed()))

    N = len(predictions)
    print('Matrix construction and multiply... ', end = '')
    with elapsed_timer() as elapsed:
        K = km.get_params(deep = False)['n_clusters']
        topics = np.zeros((K, N))
        for i, cluster in enumerate(predictions):
            topics[cluster, i] = 1.0
        topics = np.append(topics, np.full((1, N), 1.0), axis = 0)
        importance = np.append(importance, np.full((N, 1), 1.0), axis = 1)
        day = np.dot(topics, importance).flatten()
        day = np.divide(day, N)
    print('{}s'.format(elapsed()))

    with open(outfile, 'w') as o:
        writer = csv.writer(o, delimiter = '\t')
        writer.writerow(day)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
