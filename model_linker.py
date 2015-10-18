from __future__ import division
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import operator
import pandas as pd
import sys
import datetime
from itertools import izip
from sklearn.cluster import AgglomerativeClustering
import random
from scipy.cluster.vq import kmeans2, whiten
import pickle
import csv
from os import getcwd

import ipdb


def chunkwise(t, size=2):
    it = iter(t)
    return izip(*[it]*size)


# generate categorical markov array
NUM_DAYS = 200
NUM_CATEGORIES = 5


# D = sys.argv[0]  # number of dimensions of each news event
D = 2
tsne_filename = sys.argv[0]
# tsne_filename = 'all_days.out'
# hmm_filename = sys.argv[2]
# output_filename = sys.argv[3]
# price_filename = ''
output_filename = 'all_days_models.json'
all_days_file = 'all_days.csv'

# get day counts
date_counts = []
with open(all_days_file, 'r') as all_days:
    reader = csv.reader(all_days, delimiter='\t')

    last_date = -1
    date_count = 1
    for row in reader:
        date = row[0]

        if last_date == date:
            date_count += 1
        else:
            date_counts.append(date_count)
            date_count = 1

        last_date = date

ipdb.set_trace()



# read price differences
prices = []

# with open(price_filename, 'r') as csv_file:
#     reader = csv.reader(csv_file, delimiter=' ')
#     for row in reader:
price_filename = getcwd() + '/quote-download/XAG.csv'
prices = pd.read_csv(price_filename, sep=' ')
# filter prices
dates = pd.date_range('4/1/2000', periods=15)

price_range = prices[(prices.Index < '2015-04-15') & (prices.Index >= '2015-04-01')]['XAG.USD']

price_diffs = price_range.diff()
price_changes = np.array(price_diffs > 0, dtype=int)


all_floats = []
all_points = []

# read tsne data
with open(tsne_filename, 'rb') as fid:
    data_array = np.fromfile(fid, np.int16)

data_array = data_array.reshape((-1, 3))

# cluster data
NUM_CLUSTERS = 100
centroids, labels = kmeans2(whiten(data_array), NUM_CLUSTERS, iter=20)


# with open(tsne_filename, 'rb') as fid:
#     data_array = np.fromfile(fid, np.int16).reshape((-1, 2)).T

raw_news = pd.DataFrame()    # data frame holding each date's news. each column is new date
# num_events = 1306416
# num_days = 100
# date_counts = [num_events // 100] * 100
last_index = 0
date = datetime.datetime(2015,04,01)
weighted_centroids = pd.DataFrame()
for i, date_count in enumerate(date_counts):
    label_counts = np.array([0]*NUM_CLUSTERS, dtype=float)
    for data_index in range(last_index, last_index+date_count):
        label_counts[labels[data_index]] += 1
    label_counts /= sum(label_counts)
    date_string = date.strftime('%Y-%m-%d')
    day_series = pd.Series(all_points[last_index : last_index+date_count], name=date_string)
    raw_news[date_string] = day_series
    weighted_centroids[date_string] = label_counts
    date += datetime.timedelta(days=1)
    last_index += date_count


# read hmm states
hmm_states = []
# create random 2 state hmm sequence
for i in range(100):
    hmm_states.extend(np.random.dirichlet(np.ones(2), size=1))

# transform probabilistic markov states into max likelihood
prob_states = [max(enumerate(x), key=operator.itemgetter(1))[0] for x in hmm_states]


# daily_news = pd.DataFrame(daily_news)
prob_states = pd.Series(prob_states, name='prob_states')
news_df = pd.DataFrame(pd.concat([prob_states, weighted_centroids], axis=1))

# logistic regression
model1 = LogisticRegression()
# model2 = LogisticRegression()
# model3 = LogisticRegression()
# model4 = LogisticRegression()
# model5 = LogisticRegression()

# for x in set(prob_states):


# indices = news_df[news_df['prob_states'] == x].index.tolist()

# if num_days-1 in indices:
#     indices.remove(num_days-1)

# indices_next = [x+1 for x in indices]

# # features = news_df.ix[:, 1:][indices].T
# features = news_df[:][indices].T
# # output = news_df.ix[:, 0][indices_next]    
# # model1.fit(features, output)
# model1.fit(features, price_changes)


model1.fit(news_df[1:], price_changes)

with open(output_filename, 'ab+') as outf:

    pickle.dump(model1, outf)





















