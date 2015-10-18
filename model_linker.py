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


def chunkwise(t, size=2):
    it = iter(t)
    return izip(*[it]*size)


# generate categorical markov array
NUM_DAYS = 200
NUM_CATEGORIES = 5


# D = sys.argv[0]  # number of dimensions of each news event
D = 2
# tsne_filename = sys.argv[1]
tsne_filename = 'all_days.out'
# hmm_filename = sys.argv[2]
# output_filename = sys.argv[3]
price_filename = ''
output_filename = all_days_models.json


# read price differences
# prices = []
# with open(price_filename, 'r') as prices_file:
#     prices = prices_file.readlines()
prices = [random.randint(-5,5) for i in range(100)]
# prices_array = prices.split('\t')
prices_array = prices
price_delta = [x - y > 0 for x,y in zip(prices_array[1:], prices_array)]
price_changes = np.array(price_delta, dtype=int)


all_floats = []

all_points = []

# read tsne data
with open(tsne_filename, 'rb') as fid:
    data_array = np.fromfile(fid, np.int16)

all_points = zip(data_array[::2], data_array[1::2])
data_array = data_array.reshape((-1, 2))


NUM_CLUSTERS = 100
centroids, labels = kmeans2(whiten(data_array), NUM_CLUSTERS, iter=20)


# with open(tsne_filename, 'rb') as fid:
#     data_array = np.fromfile(fid, np.int16).reshape((-1, 2)).T

raw_news = pd.DataFrame()    # data frame holding each date's news. each column is new date
num_events = 1306416
num_days = 100
date_counts = [num_events // 100] * 100
last_index = 0
date = datetime.datetime(2015,03,01)
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


# generate random global new topic centroids
# NUM_CENTROIDS = 100000
# CENTROID_RANGE_RADIUS = 1000
# centroids = [(random.uniform(-CENTROID_RANGE_RADIUS, CENTROID_RANGE_RADIUS), random.uniform(-CENTROID_RANGE_RADIUS, CENTROID_RANGE_RADIUS)) for x in range(0,NUM_CENTROIDS)]
# centroids_x = [centroid[0] for centroid in centroids]
# centroids_y = [centroid[1] for centroid in centroids]


# generate news array of centroids
# NUM_DAILY_NEWS_EVENTS = 100 # num news events per day
# weight_masks = np.random.dirichlet(np.ones(NUM_CENTROIDS), size=NUM_DAYS)
# # temp_daily_news = [random.sample(centroids, NUM_DAILY_NEWS_EVENTS) for x in range(NUM_DAYS)]
# daily_news = []
# for mask in weight_masks:
#     weighted_centroids = [x for y in zip(np.multiply(mask, centroids_x), np.multiply(mask, centroids_y)) for x in y]
#     daily_news.append(weighted_centroids)

    # temp = []
    # for news_event in day:
    #     temp.append(news_event[0])
    #     temp.append(news_event[1])
    # daily_news.append(temp)

# daily_news = pd.DataFrame(daily_news)
prob_states = pd.Series(prob_states, name='prob_states')
news_df = pd.DataFrame(pd.concat([prob_states, weighted_centroids], axis=1))

# logistic regression
model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()
model4 = LogisticRegression()
model5 = LogisticRegression()

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





















