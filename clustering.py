"""
Usage: python clustering.py glob_input_array_pattern outfile K

glob_input_array_pattern should be something like "/path/to/dir/*.export.CSV", 
where you should remember to quote it when invoking the command (else the
shell will expand the glob pattern for you.

Concatenates the input numpy arrays referenced by the glob pattern,
then performs clustering, dropping the first 8 importance-related news 
data columns (according to the schema detailed in preprocessing.py and respected
 by expand.py)

Serializes (via pickle) the GMM learned based on the input to outfile.
"""

from glob import glob
from sklearn.mixture import GMM
from sklearn.preprocessing import scale
import numpy as np
import fileinput
import pickle
import sys

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1])
    outfile = sys.argv[2]
    K = int(sys.argv[3])

    print("Reading in", len(infiles), "files")
    fullarr = []
    for file_path in infiles:
        fullarr.append(
            np.genfromtxt(file_path, delimiter=','
    # fullarr = np.genfromtxt(fileinput.input(infiles), delimiter='\t', dtype='f8')

    print("Normalizing and whitening input data")
    scale(fullarr, copy = False)

    print("Training GMM with K =", K)

    gmm = GMM(n_components = K, covariance_type = 'full')
    gmm.fit(fullarr)

    cstr = "CONVERGED" if gmm.converged_ else "FAILED TO CONVERGE"
    print("GMM trained (" + cstr + "), saving.")

    with open(outfile, 'wb') as out_model:
        pickle.dump(gmm, out_model)

    print("  Means  :", gmm.means_)
    print("  Weights:", gmm.means_)
    print("  AIC    :", gmm.aic(fullarr))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



