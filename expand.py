import csv
import sys
import pickle
import numpy
from copy import copy

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
    n = int(number)
    if n:
        one_hot_array[n - 1] = 1

    return one_hot_array

input_filename = sys.argv[1]
output_filename = sys.argv[2]

input_file = open(input_filename, 'r')
# output_file = open(output_filename, 'w')
# output_writer = csv.writer(output_file, delimiter='\t')

with open('models/word2vec', 'rb') as word2vec_file:
    full_model = pickle.load(word2vec_file)
with open('models/word2vec_bigram', 'rb') as bigram_file:
    bigram = pickle.load(bigram_file)

all_lines = input_file.readlines()

NUM_STRINGS = 2
NUM_NUMERIC = 3
NUM_IMPORTANCE = 8
CATEGORICAL_TOTAL = sum(categorical_total.values())

length_of_event = 100 * NUM_STRINGS + NUM_NUMERIC + CATEGORICAL_TOTAL + NUM_IMPORTANCE
day_array = np.zeros(len(all_lines), length_of_event)

with open(input_file, 'r') as inf:
    old_field_counter = 0
    field_counter = old_field_counter
    event_counter = 0
    for event in inf:

        fields = event.split('\t')
        for i, field in enumerate(fields):
            field_type = column_types[column_idx[i]]

            combined_cameo = 0

            if field_type == 'importance':

                field_counter += 1
                day_array[event_counter][old_field_counter:field_counter] = field


            elif field_type == 'categorical':
                category = column_idx[i]

                if category == 'CAMEOCode2' or category == 'CAMEOCode3':
                    continue

                category_total = categorical_total[category]
                one_hot_array = one_hot(field, category_total)

                field_counter += len(one_hot_array)
                day_array[event_counter][old_field_counter:field_counter] = copy(one_hot_array)

            elif field_type == 'string':
                split_field = field.split(' ')
                if len(split_field) == 1:
                    try:
                        word_vec = full_model[split_field]
                    except:
                        continue
                elif len(split_field) == 2:
                    try:
                        word_vec = full_model[bigram[split_field]]
                    except:
                        continue
                else: # TODO can add trigrams if needed
                    continue

                field_counter += len(word_vec)

                day_array[event_counter][old_field_counter:field_counter] = copy(word_vec)

            else:
                field_counter += 1
                day_array[event_counter][old_field_counter:field_counter] = field

            old_field_counter = field_counter

np.savetxt(output_filename, day_array, delimiter='\t')















