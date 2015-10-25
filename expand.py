"""
Usage: python expand.py infile outfile [models_dir]

Loads word2vec model from models_dir directory from pickle file of the same
name. Similarly for word2vec_(bi|tri|quad)gram. Default models_dir is
/n/fs/gcf/COS513-Finance/models.

Translates the schema outputted from preprocessing.py to a numerical-only
float array, printed in csv format to outfile.
"""
import csv
import sys
import pickle
import time
import os
import numpy as np

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

column_names = ['DaysSincePublished', 'IsVerbal', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'CAMEOCode1', 'CAMEOCodeFull', 'IsCooperative', 'Actor1Country', 'Actor2Country', 'Actor1Geo_Type', 'Actor2Geo_Type', 'ActionGeo_Type', 'ActionGeo_Lat', 'ActionGeo_Long', 'Actor1Name', 'Actor2Name', ]

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

            # TODO: the current model is missing some really obvious stuff,
            # like singapore and protester. We need to significantly expand our
            # corpus. For now, we don't even use actor names.
            
            field = field.strip()
            split_field = field.split(' ')
            try:
                if len(field) == 0:
                    word_vec = [0 for i in range(100)]
                elif len(split_field) == 1:
                    word_vec = full_model[split_field].tolist()[0]
                elif len(split_field) == 2:
                    word_vec = full_model[bigram[split_field]].tolist()[0]
                elif len(split_field) == 3:
                    word_vec = full_model[trigram[bigram[split_field]]].tolist()[0]
                else:
                    word_vec = full_model[quadgram[trigram[bigram[split_field]]]].tolist()[0]
            except KeyError:
                return None
            expanded.extend(word_vec)
        else:
            expanded.append(field)
    expanded.extend(importance)
    return expanded

def load_models(models_dir):
    print('Loading models... ', end='')
    with elapsed_timer() as elapsed:
        with open(os.path.join(models_dir, 'word2vec'), 'rb') as word2vec_file:
            full_model = pickle.load(word2vec_file)
        with open(os.path.join(models_dir, 'word2vec_bigram'), 'rb') as b:
            bigram = pickle.load(b)
        with open(os.path.join(models_dir, 'word2vec_trigram'), 'rb') as t:
            trigram = pickle.load(t)
        with open(os.path.join(models_dir, 'word2vec_quadgram'), 'rb') as q:
            quadgram = pickle.load(q)
    print('{}s'.format(elapsed()))
    return (full_model, bigram, trigram, quadgram)


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(__doc__)
        return 1

    infile = sys.argv[1]
    outfile = sys.argv[2]
    models_dir = '/n/fs/gcf/COS513-Finance/models'
    if len(sys.argv) == 4:
        models_dir = sys.argv[3]

    with open(infile, 'r') as i, open(outfile, 'w') as o:
        reader = csv.reader(i, delimiter = '\t')
        writer = csv.writer(o, delimiter = '\t')
        full_model, bigram, trigram, quadgram = load_models(models_dir)
        
        tot_rows = 0
        dropped_rows = 0
        print('Expanding rows... ', end = '')
        with elapsed_timer() as elapsed:
            for fields in reader:
                expanded = expand_row(fields)
                if expanded:
                    writer.writerow(expanded)
                else:
                    dropped_rows += 1
                tot_rows += 1
    print('{}s'.format(elapsed()))
    print('Dropped {} of {}'.format(dropped_rows, tot_rows))
    tot_cols = sum((categorical_total[col] if coltype == 'categorical' else
                    0 if coltype == 'string' else # TODO rm once we use w2v
                    1 for col, coltype in column_types.items()))
    imp_cols = list(column_types.values()).count('importance')
    print('Total columns: {}, last {} are importance'.format(tot_cols, imp_cols))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
