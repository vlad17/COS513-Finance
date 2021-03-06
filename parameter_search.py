"""
Usage: python parameter_search.py infile max_components num_random_lines
Example: python parameter_search.py /n/fs/gcf/dchouren-repo/COS513-Finance/random_10k_expanded 10000 10000

Infile should be an csv file of expanded rows

Parameter searches for alpha and prints best alpha with its score
"""

import numpy as np
import fileinput
import sys
from sklearn.mixture import DPGMM

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    infile = sys.argv[1]
    N = int(sys.argv[2])    
    num_random = int(sys.argv[3])

    print("Reading in", infile)
    fullarr = np.loadtxt(fileinput.input(infile), delimiter = '\t')[:,:-7]

    stds = np.apply_along_axis(np.std, 0, fullarr)[:,np.newaxis].T
    means = np.apply_along_axis(np.mean, 0, fullarr)[:,np.newaxis].T
    stds[stds == 0] = 1.0

    num_lines = num_random
    fullarr = fullarr[np.random.choice(fullarr.shape[0], num_lines, replace=True),:]

    fullarr = (fullarr - means) / stds

    output = ''

    print("Parameter searching...")
    igmm = None
    best_score = -100000
    best_alpha = -1
    best_model = None
    for alpha in [0.01,0.1,1,10]: 
        print("Learning infinite GMM with N={}, alpha={}".format(N, alpha))
        output += "Learning infinite GMM with N={}, alpha={}\n".format(N, alpha)
        igmm = DPGMM(covariance_type='diag', n_components=N, alpha=alpha, init_params='wmc')
        igmm.fit(fullarr)
        score = igmm.score(fullarr)
        score = sum(score)/len(score)
        print('{}: {} with {} clusters'.format(alpha, score, igmm.n_components))
        output += '{}: {} with {} clusters\n'.format(alpha, score, igmm.n_components)

        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_model = igmm

    print('Best alpha={}, score={}'.format(best_alpha, best_score))
    output += 'Best alpha={}, score={}\n'.format(best_alpha, best_score)
    with open('parameter_search_results.txt', 'a+') as outf:
        outf.write(output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())