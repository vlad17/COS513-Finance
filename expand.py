import csv
import sys
import pickle
import time
import numpy as np

column_names = ['EventDaysSinceEpoch', 'PublishedDaysSinceEpoch', 'IsVerbal', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'CAMEOCode1', 'CAMEOCodeFull', 'IsCooperative', 'Actor1Country', 'Actor2Country', 'Actor1Geo_Type', 'Actor2Geo_Type', 'ActionGeo_Type', 'ActionGeo_Lat', 'ActionGeo_Long', 'Actor1Name', 'Actor2Name', ]

column_idx = dict(zip(range(len(column_names)), column_names))

column_types = {
    'EventDaysSinceEpoch': 'importance',
    'PublishedDaysSinceEpoch': 'importance',
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

importance_idx = {
    'EventDaysSinceEpoch': 0,
    'PublishedDaysSinceEpoch': 1,
    'IsVerbal': 2,
    'GoldsteinScale': 3,
    'NumMentions': 4,
    'NumSources': 5,
    'NumArticles': 6,
    'AvgTone': 7,
}
categorical_idx = {
    'CAMEOCode1': 8,        # 1-20
    'CAMEOCode2': 9,        # 1-100
    'CAMEOCode3': 10,       # 1-10
    'Actor1Country': 12,    # 1-1000
    'Actor2Country': 13,    # 1-1000
    'Actor1Geo_Type': 14,   # 1-5
    'Actor2Geo_Type': 15,   # 1-5
    'ActionGeo_Type': 16,   # 1-1000
}
numeric_idx = {
    'IsCooperative': 11,
    'ActionGeo_Lat': 17,
    'ActionGeo_Long': 18,
}
string_idx = {
    'Actor1Name': 19,
    'Actor2Name': 20,
}

categorical_total = {
    'CAMEOCode1': 20,
    'CAMEOCodeFull': 350,
    'Actor1Country': 1000,
    'Actor2Country': 1000,
    'Actor1Geo_Type': 5,
    'Actor2Geo_Type': 5,
    'ActionGeo_Type': 1000,
}

def one_hot(number, total):
    one_hot_array = [0 for i in range(total)]
    one_hot_array[int(number)-1] = 1

    return one_hot_array


input_filename = sys.argv[1]
output_filename = sys.argv[2]

input_file = open(input_filename, 'r')
output_file = open(output_filename, 'w')
output_writer = csv.writer(output_file, delimiter='\t')

start_time = time.time()
full_model = bigram = trigram = quadgram = None
with open('models/word2vec', 'rb') as word2vec_file:
    full_model = pickle.load(word2vec_file)
with open('models/word2vec_bigram', 'rb') as bigram_file:
    bigram = pickle.load(bigram_file)
with open('models/word2vec_trigram', 'rb') as bigram_file:
    trigram = pickle.load(bigram_file)
with open('models/word2vec_quadgram', 'rb') as bigram_file:
    quadgram = pickle.load(bigram_file)

all_data = []

start_time = time.time()
tot_rows = 0
dropped_rows = 0
for event in input_file.readlines():
    expanded = []
    importance = []

    fields = event.split('\t')
    drop_line = False
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
                drop_line = True
                continue
            expanded.extend(word_vec)
        else:
            expanded.append(field)
    expanded.extend(importance)

    if drop_line:
        dropped_rows += 1
    else:
        output_writer.writerow(expanded)
    tot_rows += 1

print("Dropped", dropped_rows, "of", tot_rows)

input_file.close()
output_file.close()
