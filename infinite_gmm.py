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

Example: python infinite_gmm.py /n/fs/gcf/dchouren-repo/COS513-Finance/igmm100k_2.export.CSV* /n/fs/gcf/dchouren-repo/COS513-Finance/models/igmm_alpha 1000 0.6
"""

from glob import glob
import numpy as np
import fileinput
import pickle
import sys
from sklearn.mixture import DPGMM
import itertools
# import ipdb

TOPIC_COLUMNS = 938 

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1])
    outfile = sys.argv[2]
    N = int(sys.argv[3])
    alpha = float(sys.argv[4])

    print("Reading in", len(infiles), "files")
    fullarr = np.loadtxt(fileinput.input(infiles), delimiter = '\t')[:,:-7]


    stds = np.apply_along_axis(np.std, 0, fullarr)[:,np.newaxis].T
    means = np.apply_along_axis(np.mean, 0, fullarr)[:,np.newaxis].T
    stds[stds == 0] = 1.0

    num_lines = 10000
    fullarr = fullarr[np.random.choice(fullarr.shape[0], num_lines, replace=True),:]

    fullarr = (fullarr - means) / stds


    print("Learning infinite GMM with N={}, alpha={}".format(N, alpha))

    igmm = DPGMM(covariance_type='diag', n_components=N, alpha=alpha, init_params='wmc')
    igmm.fit(fullarr)

    print("Infinite GMM trained, saving")

    with open(outfile + '_' + num_lines, 'wb') as out_model:
        pickle.dump(igmm, out_model)

    print("Score:", igmm.score(fullarr))
    print("Num Components:", igmm.n_components)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



