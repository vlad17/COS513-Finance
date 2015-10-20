from __future__ import division
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import operator
import pandas as pd
import sys
import datetime
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.vq import kmeans2, whiten
import pickle
import csv
from os import getcwd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import ipdb


# D = sys.argv[0]  # number of dimensions of each news event
D = 3
# tsne_filename = sys.argv[0]
tsne_filename = 'data/theta-8-no_dims-3-perplexity-17-rand-73.out'

# hmm_filename = sys.argv[2]
# output_filename = sys.argv[3]
output_filename = 'all_days_models.json'


#####################################################################################################

start_date = '2015-04-01'
end_date = '2015-04-15'
all_days_file = 'data/all_days.csv'   # used to grab day counts

def get_event_counts_by_date(event_log_file, start_date, end_date):
    """ Trawl event log to get event counts by date for given range """

    valid_indices = []

    EPOCH = datetime.datetime(1970,1,1)
    start_since_epoch = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - EPOCH).days
    end_since_epoch = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - EPOCH).days

    date_counts = [0 for counter in range(end_since_epoch - start_since_epoch)]

    with open(event_log_file, 'r') as event_log:
        reader = csv.reader(event_log, delimiter='\t')

        for i, row in enumerate(reader):
            date = int(row[0])
            diff_from_start = date - start_since_epoch
            if diff_from_start in range(end_since_epoch - start_since_epoch):
                date_counts[diff_from_start] += 1

                valid_indices.append(i)


    return date_counts, valid_indices


date_counts, valid_indices = get_event_counts_by_date(all_days_file, start_date, end_date)
# date_counts = np.array(date_counts) * 4

#####################################################################################################

# read price differences
price_filename = 'quote-download/XAG.csv'
commodity = 'XAG.USD'

def get_price_changes(price_filename, commodity, start_date, end_date):
    """ Get changes in price for a commodity over given range """
    prices = []
    prices = pd.read_csv(price_filename, sep=' ')

    price_range = prices[(prices['Index'] < '2015-04-15') & (prices['Index'] >= '2015-04-01')][commodity]
    price_diffs = price_range.diff()
    price_changes = np.array(price_diffs > 0, dtype=int)

    return price_changes

price_changes = get_price_changes(price_filename, commodity, start_date, end_date)

#####################################################################################################

# read tsne data
with open(tsne_filename, 'rb') as fid:
    data_array = np.fromfile(fid, np.int64)

data_array = data_array.reshape((-1, D))[valid_indices]

# cluster data
NUM_CLUSTERS = 100
centroids, labels = kmeans2(whiten(data_array), NUM_CLUSTERS, iter=20)

# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# ax.scatter(data_array[:, 2], data_array[:, 0], data_array[:, 1], c=labels.astype(np.float))

# plt.show()

#####################################################################################################

def get_weighted_centroids(news_events, date_counts, start_date):
    # raw_news = pd.DataFrame() 
    date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    weighted_centroids = pd.DataFrame()
    last_index = 0

    for i, date_count in enumerate(date_counts):
        label_counts = np.array([0]*NUM_CLUSTERS, dtype=float)

        for data_index in range(last_index, last_index+date_count):
            label_counts[labels[data_index]] += 1

        label_counts /= sum(label_counts)
        date_string = date.strftime('%Y-%m-%d')

        # day_series = pd.Series(data_array[last_index : last_index+date_count], name=date_string)

        # raw_news[date_string] = day_series
        weighted_centroids[date_string] = label_counts

        date += datetime.timedelta(days=1)
        last_index += date_count

    return weighted_centroids

weighted_centroids = get_weighted_centroids(data_array, date_counts, start_date)

#####################################################################################################

# read hmm states
hmm_states = []
# create random 2 state hmm sequence
for i in range(100):
    hmm_states.extend(np.random.dirichlet(np.ones(2), size=1))

# transform probabilistic markov states into max likelihood
prob_states = [max(enumerate(x), key=operator.itemgetter(1))[0] for x in hmm_states]

prob_states = pd.Series(prob_states, name='prob_states')

#####################################################################################################

news_df = pd.DataFrame(pd.concat([prob_states, weighted_centroids], axis=1))
news_df_t = news_df.T

train = news_df_t[:-4]
test = news_df_t[-4:]

# logistic regression
model1 = LogisticRegression()

model1.fit(train, price_changes[:-3])
model1.score(test, price_changes[-4:])

with open(output_filename, 'ab+') as outf:
    pickle.dump(model1, outf)





















