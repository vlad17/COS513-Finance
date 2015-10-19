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
# tsne_filename = sys.argv[0]
tsne_filename = 'data/theta-1.1-no_dims-3-perplexity-3-rand-73.out'
date_start = '2015-04-01'
date_end = '2015-04-15'
# tsne_filename = 'all_days.out'
# hmm_filename = sys.argv[2]
# output_filename = sys.argv[3]
# price_filename = ''
output_filename = 'all_days_models.json'
all_days_file = 'data/all_days.csv'

epoch = datetime.datetime(1970,1,1)
start_since_epoch = datetime.datetime.strptime(date_start, '%Y-%m-%d') - epoch
end_since_epoch = datetime.datetime.strptime(date_end, '%Y-%m-%d') - epoch

# get day counts
date_counts = [0 for i in range((end_since_epoch - start_since_epoch).days)]

with open(all_days_file, 'r') as all_days:
    reader = csv.reader(all_days, delimiter='\t')

    for row in reader:
        date = int(row[0])
        diff_from_start = date - start_since_epoch.days
        if diff_from_start in range((end_since_epoch - start_since_epoch).days):
            date_counts[diff_from_start] += 1

# ipdb.set_trace()



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

model1.fit(news_df.T[1:], price_changes)

with open(output_filename, 'ab+') as outf:

    pickle.dump(model1, outf)





















