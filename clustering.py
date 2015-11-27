
"""
Usage: python clustering.py glob_input_array_pattern outfile K

glob_input_array_pattern should be something like "/path/to/dir/*.export.CSV", 
where you should remember to quote it when invoking the command (else the
shell will expand the glob pattern for you.

Concatenates the input numpy arrays referenced by the glob pattern,
then performs clustering, dropping the last 8 importance-related news 
data columns (according to the schema detailed in preprocessing.py and respected
 by expand.py)

Serializes (via pickle) the GMM learned based on the input to outfile.
"""

from glob import glob
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import itertools
import fileinput
import pickle
import sys

TOPIC_COLUMNS = 1138 

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1])
    outfile = sys.argv[2]
    K = int(sys.argv[3])

    print("Reading in", len(infiles), "files")
    fullarr = np.loadtxt(fileinput.input(infiles), delimiter = '\t')[:,7:]
    print(infiles[0])
    # if infiles[0] == 'random_sample_20000101':
    print("Normalizing")
    stds = np.apply_along_axis(np.std, 0, fullarr)[:,np.newaxis].T
    means = np.apply_along_axis(np.mean, 0, fullarr)[:,np.newaxis].T
    standardized_arr = (fullarr - means) / stds

    with open('/n/fs/gcf/dchouren-repo/COS513-Finance/summary_stats/stats', 'wb+') as summary_stats_outf:
        np.savetxt(summary_stats_outf, stds.T, delimiter='\t')
    with open('/n/fs/gcf/dchouren-repo/COS513-Finance/summary_stats/stats', 'ab') as summary_stats_outf:
        np.savetxt(summary_stats_outf, means.T, delimiter='\t')


    print("Learning MiniBatchKMeans with K =", K)

    km = MiniBatchKMeans(n_clusters = K, verbose = True) # TODO max_iter
    km.fit(standardized_arr)

    print("KMeans trained, saving")

    with open(outfile, 'wb') as out_model:
        pickle.dump(km, out_model)

    print("Score:", km.score(standardized_arr))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



