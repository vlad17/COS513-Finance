
"""
Usage: python examine_clusters.py raw_data_dir model_file 

Examines cluster model for a given K

Example: python examine_clusters.py /n/fs/gcf/raw-data /n/fs/scratch/dchouren/models/1000.model
"""

from glob import glob
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import itertools
import fileinput
import pickle
import sys
import os
import preprocessing
# from expand import expand_row

TOPIC_COLUMNS = 938 

############## EXPAND STUFF ##############

column_names = ['DaysSincePublished', 'IsVerbal', 'GoldsteinScale', 
                'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 
                'CAMEOCode1', 'CAMEOCodeFull', 'IsCooperative', 
                'Actor1Country', 'Actor2Country', 'Actor1Geo_Type', 
                'Actor2Geo_Type', 'ActionGeo_Type', 'ActionGeo_Lat', 
                'ActionGeo_Long', 'Actor1Name', 'Actor2Name']

column_idx = dict(zip(range(len(column_names)), column_names))

column_types = {
    'DaysSincePublished': 'importance',
    'IsVerbal': 'importance',
    'GoldsteinScale': 'importance',
    'NumMentions': 'importance',
    'NumSources': 'importance',
    'NumArticles': 'importance',
    'AvgTone': 'importance',
    'CAMEOCode1': 'categorical',
    'CAMEOCodeFull': 'categorical',
    'IsCooperative': 'numeric',
    'Actor1Country': 'categorical',
    'Actor2Country': 'categorical',
    'Actor1Geo_Type': 'categorical',
    'Actor2Geo_Type': 'categorical',
    'ActionGeo_Type': 'categorical',
    'ActionGeo_Lat': 'numeric',
    'ActionGeo_Long': 'numeric',
    'Actor1Name': 'string',
    'Actor2Name': 'string',
}

categorical_total = {
    'CAMEOCode1': 20,
    'CAMEOCodeFull': 350,
    'Actor1Country': 275,
    'Actor2Country': 275,
    'Actor1Geo_Type': 5,
    'Actor2Geo_Type': 5,
    'ActionGeo_Type': 5,
}

def one_hot(number, total):
    one_hot_array = [0 for i in range(total)]
    one_hot_array[int(number)-1] = 1

    return one_hot_array


def expand_row(fields):
    expanded = []
    importance = []
    for i, field in enumerate(fields):

        field_type = column_types[column_idx[i]]

        if field_type == 'importance':
            importance.append(field)

        elif field_type == 'categorical':
            category = column_idx[i]

            category_total = categorical_total[category]
            one_hot_array = one_hot(field, category_total)

            expanded.extend(one_hot_array)

        elif field_type == 'string':
            continue

        else:
            expanded.append(field)
    expanded.extend(importance)
    return expanded



def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    infiles = glob(sys.argv[1] + '/*')

    print("Loading model")
    model = None
    with open(sys.argv[2], 'rb') as model_file:
        model = pickle.load(model_file)

    tmp_file = '/tmp/random_lines_for_clusting'
    try:
        os.remove(tmp_file)
    except OSError:
        pass

    # take num_lines lines from each file
    num_lines = 10
    for inf in infiles:
        os.system('shuf -n {} {} >> {}'.format(num_lines, inf, tmp_file))


    with open(tmp_file) as random_lines_file:
        for line in random_lines_file:
            print(line)
            preprocessed = preprocessing.clean_row(line)
            # fields = preprocessed.split('\t')
            expanded = expand_row(preprocessed)

            print(expanded)

            print()
            print()


    # print("Reading in", len(infiles), "files")
    # fullarr = np.loadtxt(fileinput.input(infiles), delimiter = '\t')

    # print("Learning MiniBatchKMeans with K =", K)

    # km = MiniBatchKMeans(n_clusters = K, verbose = True) # TODO max_iter
    # km.fit(fullarr)

    # print("KMeans trained, saving")

    # with open(outfile, 'wb') as out_model:
    #     pickle.dump(km, out_model)

    # print("Score:", km.score(fullarr))
    
    # return 0

if __name__ == "__main__":
    sys.exit(main())



