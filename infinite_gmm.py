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
"""

from glob import glob
import numpy as np
import fileinput
import pickle
import sys
from sklearn.mixture import DPGMM

TOPIC_COLUMNS = 1138 

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1])
    outfile = sys.argv[2]
    N = int(sys.argv[3])
    alpha = int(sys.argv[4])

    print("Reading in", len(infiles), "files")
    fullarr = np.loadtxt(fileinput.input(infiles), delimiter = '\t',
                         usecols = range(TOPIC_COLUMNS))

    print("Learning infinite GMM with N={}, alpha={}".format(N, alpha))

    igmm = DPGMM(n_components=N, alpha=alpha, init_params='wmc', verbose=True)
    igmm.fit(fullarr)

    print("Infinite GMM trained, saving")

    with open(outfile, 'wb') as out_model:
        pickle.dump(igmm, out_model)

    print("Score:", igmm.score(fullarr))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



