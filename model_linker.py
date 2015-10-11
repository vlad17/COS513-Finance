import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import operator
import pandas as pd


# generate categorical markov array
NUM_DAYS = 200
NUM_CATEGORIES = 5

fake_hmm = np.random.dirichlet(np.ones(NUM_CATEGORIES), size=NUM_DAYS)


# transform probabilistic markov states into max likelihood
prob_states = [max(enumerate(x), key=operator.itemgetter(1))[0] for x in fake_hmm]


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
    weighted_centroids = zip(np.multiply(mask, centroids_x), np.multiply(mask, centroids_y))
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
    this_prob_states = news_df[news_df['prob_states'] == x]
    
model.fit(this_prob_states.ix[:,1:], this_prob_states.ix[:,0])

print model.coef_





















