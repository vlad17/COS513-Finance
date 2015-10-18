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


all_floats = []

all_points = []

# read tsne data
with open(tsne_filename, 'r') as input_file:
    all_input = input_file.readlines()
    all_floats = all_input[0].split("\t")

# all_points = chunkwise(all_floats, 2)
all_points = zip(all_floats[::2], all_floats[1::2])


raw_news = pd.DataFrame()    # data frame holding each date's news. each column is new date
num_events = 107236
num_days = 100
date_counts = [107236 // 100] * 100
last_index = 0
date = datetime.datetime(2015,03,01)
for i, date_count in enumerate(date_counts):

    date_string = date.strftime('%Y-%m-%d')
    day_series = pd.Series(all_points[last_index: last_index+date_count], name=date_string)

    raw_news[date_string] = day_series

    date += datetime.timedelta(days=1)
    last_index = last_index + date_count


# read hmm states
hmm_states = []
# create random 2 state hmm sequence
for i in range(100):
    hmm_states.extend([random.randint(0,1)])


# transform probabilistic markov states into max likelihood
prob_states = [max(enumerate(x), key=operator.itemgetter(1))[0] for x in hmm_states]


# generate random global new topic centroids
NUM_CENTROIDS = 100000
CENTROID_RANGE_RADIUS = 1000
centroids = [(random.uniform(-CENTROID_RANGE_RADIUS, CENTROID_RANGE_RADIUS), random.uniform(-CENTROID_RANGE_RADIUS, CENTROID_RANGE_RADIUS)) for x in range(0,NUM_CENTROIDS)]
centroids_x = [centroid[0] for centroid in centroids]
centroids_y = [centroid[1] for centroid in centroids]


# generate news array of centroids
NUM_DAILY_NEWS_EVENTS = 100 # num news events per day
weight_masks = np.random.dirichlet(np.ones(NUM_CENTROIDS), size=NUM_DAYS)
# temp_daily_news = [random.sample(centroids, NUM_DAILY_NEWS_EVENTS) for x in range(NUM_DAYS)]
daily_news = []
for mask in weight_masks:
    weighted_centroids = [x for y in zip(np.multiply(mask, centroids_x), np.multiply(mask, centroids_y)) for x in y]
    daily_news.append(weighted_centroids)

    # temp = []
    # for news_event in day:
    #     temp.append(news_event[0])
    #     temp.append(news_event[1])
    # daily_news.append(temp)

daily_news = pd.DataFrame(daily_news)
prob_states = pd.Series(prob_states, name='prob_states')
news_df = pd.DataFrame(pd.concat([prob_states, daily_news], axis=1))

# logistic regression
model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()
model4 = LogisticRegression()
model5 = LogisticRegression()

for x in set(prob_states):
    indices = [news_df['prob_states'][:-1] == x]    # go up to second to last day
    indices_next = [x+1 for x in indices]

    features = news_df[indices].ix[:, 1:]
    output = news_df[indices_next].ix[:, 0]
    
# model1.fit(this_prob_states.ix[:, 1:][1:], this_prob_states.ix[:, 0][:-1])





















