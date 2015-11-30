"""
Usage: python infinite_gmm.py glob_input_array_pattern outfile N alpha

glob_input_array_pattern should be something like "/path/to/dir/*.export.CSV", 
where you should remember to quote it when invoking the command (else the
shell will expand the glob pattern for you.

N is the max number of components
alpha is a hyperparameter for the number of components

Concatenates the input numpy arrays referenced by the glob pattern,
then runs them through a DPGMM and prints the score.

Serializes (via pickle) the GMM learned based on the input to outfile.

Example: python infinite_gmm.py /n/fs/gcf/dchouren-repo/COS513-Finance/igmm_10000_20000101.export.CSV* /n/fs/gcf/dchouren-repo/COS513-Finance/models/igmm_alpha 10000 1
"""

from glob import glob
import numpy as np
import fileinput
import pickle
import sys
from sklearn.mixture import DPGMM
import itertools
import ipdb

TOPIC_COLUMNS = 938 

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1])
    outfile = sys.argv[2]
    N = int(sys.argv[3])
    alpha = int(sys.argv[4])

    print("Reading in", len(infiles), "files")
    fullarr = np.loadtxt(fileinput.input(infiles), delimiter = '\t')

    print("Parameter searching...")
    igmm = None
    best_score = -1
    best_alpha = 
    best_model = None
    for alpha in itertools.chain(np.arange(0.1,1,0.1), np.arange(1,10,1)): 
        print("Learning infinite GMM with N={}, alpha={}".format(N, alpha))
        igmm = DPGMM(covariance_type='diag', n_components=N, alpha=alpha, init_params='wmc')
        igmm.fit(fullarr)
        score = igmm.score(fullarr)
        score = sum(score)/len(score)
        print(score)

        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_model = igmm

    print("Best alpha={}".format(best_alpha))


    # print("Learning infinite GMM with N={}, alpha={}".format(N, alpha))

    # igmm = DPGMM(n_components=N, alpha=alpha, init_params='wmc', verbose=True)
    # igmm.fit(fullarr)

    print("Infinite GMM trained, saving")

    with open(outfile, 'wb') as out_model:
        pickle.dump(best_model, out_model)

    print("Score:", best_model.score(fullarr))
    print("Num Components:", igmm.n_components)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



