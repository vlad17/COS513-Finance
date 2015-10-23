from gensim.models import Word2Vec, Phrases
from nltk.corpus import brown
import gensim
import csv
import sys

column_names = ['EventDaysSinceEpoch', 'PublishedDaysSinceEpoch', 'IsVerbal', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'CAMEOCode1', 'CAMEOCode2', 'CAMEOCode3', 'CAMEOCode', 'IsCooperative', 'Actor1Country', 'Actor2Country', 'Actor1Geo_Type', 'Actor2Geo_Type', 'ActionGeo_Type', 'ActionGeo_Lat', 'ActionGeo_Long', 'Actor1Name', 'Actor2Name', ]

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
    'CAMEOCode2': 'categorical',
    'CAMEOCode3': 'categorical',
    'CAMEOCode': 'categorical',
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
    'CAMEOCode': 350,
    'Actor1Country': 1000,
    'Actor2Country': 1000,
    'Actor1Geo_Type': 5,
    'Actor2Geo_Type': 5,
    'ActionGeo_Type': 1000,
}

def one_hot(number, total):
    one_hot_array = [0 for i in range(total)]
    one_hot_array[number-1] = 1

    return one_hot_array


input_filename = sys.argv[0]
output_filename = sys.argv[1]

input_file = open(input_filename, 'r')
output_file = open(output_filename, 'w')
output_writer = csv.writer(output_file, delimiter='\t')

# set up word2vec
brown_sents = brown.sents()
brown_lower = []
for sentence in brown_sents:
    sentence = [word.lower() for word in sentence]
    brown_lower.append(sentence)

bigram = Phrases(brown_lower)
full_model = gensim.models.Word2Vec(bigram[brown_lower], min_count=1)


for event in input_file.readlines():
    expanded = []
    importance = []

    fields = event.split('\t')
    for i, field in enumerate(fields):
        field_type = column_types[column_idx[i]]

        combined_cameo = 0

        if field_type == 'importance':
            importance.append(field)

        elif field_type == 'categorical':
            category = column_idx[i]

            if category == 'CAMEOCode2' or category == 'CAMEOCode3':
                continue

            category_total = categorical_total[category]
            one_hot_array = one_hot(field, category_total)

            expanded.extend(one_hot_array)

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

            expanded.extend(word_vec)

        else:
            expanded.append(field)

    expanded.extend(importance)

    output_writer.writerow(expanded)

    input_file.close()
    output_file.close()














