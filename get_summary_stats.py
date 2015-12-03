"""
Usage: python get_summary_stats.py infile outfile 
Example: python get_summary_stats.py /n/fs/gcf/dchouren-repo/COS513-Finance/random_20000101.export.CSV /n/fs/gcf/dchouren-repo/COS513-Finance/summary_stats/stats
"""

import numpy as np
import sys
import fileinput

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1

    infile = sys.argv[1]
    outfile = sys.argv[2]

    print("Reading in", infile)
    fullarr = np.loadtxt(fileinput.input(infile), delimiter = '\t')[:,:-7]
    
    stds = np.apply_along_axis(np.std, 0, fullarr)[:,np.newaxis].T
    means = np.apply_along_axis(np.mean, 0, fullarr)[:,np.newaxis].T

    stds[stds == 0] = 1.0
    
    with open(outfile, 'wb+') as summary_stats_outf:
        np.savetxt(summary_stats_outf, stds, delimiter='\t')
    with open(outfile, 'ab') as summary_stats_outf:
        np.savetxt(summary_stats_outf, means, delimiter='\t')


    return 0

if __name__ == "__main__":
    sys.exit(main())