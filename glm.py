"""
Usage: python glm.py 
# TODO: change args to take train_start, train_end, test_start, test_end, commodity name, K

Loads expanded, summarized feature set of K days, splits data into training and test sets, and trains various logistic regression models on the training sets
Selects best parameters based on regularization errors

"""

import numpy as np
import pandas as pd
import glob
import fileinput
from sklearn.linear_model import LogisticRegression
import pickle
import itertools
import sys
from sklearn.metrics import *
from pprint import pprint
# import matplotlib as plt

k = sys.argv[1]

def get_price_info(price_filename, commodity):
    """ Get changes in price for a commodity over given range """
    prices = []
    prices = pd.read_csv(price_filename, sep=' ', index_col=0)

    five_day_avg = pd.Series(pd.rolling_mean(prices[commodity], 5), name='five_day_avg')
    ten_day_avg = pd.Series(pd.rolling_mean(prices[commodity], 10), name='ten_day_avg')
    thirty_day_avg = pd.Series(pd.rolling_mean(prices[commodity], 30), name='thirty_day_avg')
    
    price_info = pd.DataFrame(pd.concat([prices, five_day_avg, ten_day_avg, thirty_day_avg], axis=1))
    
    price_diffs = prices[commodity].diff()
    price_changes_series = pd.Series(np.array(price_diffs > 0), dtype=int, index=price_info.index.values)

    return price_info, price_changes_series


def main():    

    summarized_dir = '../CORRECT-summary-data-20130401-20151021/' + k + '/'

    # print summarized_dir
    summarized_files = glob.glob(summarized_dir + '*.csv')
    # print summarized_files[0]
    print('Reading {} files'.format(len(summarized_files)))

    train_start = '2013-04-01'
    train_end = '2014-09-30'
    valid_start = '2014-10-01'
    valid_end = '2014-11-30'
    test_start = '2014-12-01'
    test_end = '2015-02-08'

    dates = []
    for sfile in summarized_files:
        date_segment = sfile.split('/')[-1].split('.')[0]
        dates.append(date_segment[:4] + '-' + date_segment[4:6] + '-' + date_segment[6:])

    all_days_array = np.loadtxt(fileinput.input(summarized_files), delimiter = '\t')
    all_days = pd.DataFrame(all_days_array, index=dates)


    # read price differences
    commodity = 'XAG'
    price_filename = 'quote-download/' + commodity + '.csv'
    commodity = commodity + '.USD'


    price_info, price_changes = get_price_info(price_filename, commodity)
    price_info_slice = price_info[price_info.index.isin(dates)]
    price_changes = price_changes[price_changes.index.isin(dates)]

    all_features = all_days.join(price_info_slice)

    train = all_features[(all_features.index >= train_start) & (all_features.index <= train_end)]
    train_y = price_changes[(price_changes.index >= train_start) & (price_changes.index <= train_end)]

    valid = all_features[(all_features.index >= valid_start) & (all_features.index <= valid_end)]
    valid_y = price_changes[(price_changes.index >= valid_start) & (price_changes.index <= valid_end)]

    test = all_features[(all_features.index >= test_start) & (all_features.index <= test_end)]
    test_y = price_changes[(price_changes.index >= test_start) & (price_changes.index <= test_end)]


    use_dual = train.shape[0] < train.shape[1]
    best = None
    best_score = 0
    for reg in ['l1', 'l2']:
        # choose regularization value based on validation error
        for c in itertools.chain(np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1),
                                 np.arange(1, 10, 1)):
            model = LogisticRegression(penalty = reg, C = c, tol = 0.0001, 
                                       dual = use_dual and reg == 'l2')
            model.fit(train, train_y)
            score = model.score(valid, valid_y)
            prob = model.predict_proba(valid)
            if score > best_score:
                best = model
                best_score = score
    reg = best.get_params(deep = False)['penalty']
    c = best.get_params(deep = False)['C']

    #train = pd.concat([train, valid])
    #train_y = pd.concat([train_y, valid_y])

    #best.fit(train, train_y)
    print('------------------------')
    print('K: {}'.format(k))
    print('Model (reg = {}, c = {}) training acc: {} test accuracy: {}'.format(
        reg, c, best.score(train, train_y), best.score(test, test_y)))

    # find precision/recall for cutoffs 
    cutoffs = [0.1, 0.25, 0.5, 0.75, 0.9]
    all_precisions = {}
    all_recalls = {}
    all_f_scores = {}
    for cutoff in cutoffs:
        proba0 = best.predict_proba(test)[:,0]   # probability we predict first class (0)
        cutoff_preds = np.array(proba0 > cutoff, dtype=int)

        avg_precision = average_precision_score(test_y, cutoff_preds, average='weighted')
        all_precisions[cutoff] = avg_precision

        avg_recall = recall_score(test_y, cutoff_preds, average='weighted')
        all_recalls[cutoff] = avg_recall

        f_score = f1_score(test_y, cutoff_preds)
        all_f_scores[cutoff] = f_score

    print
    print('Precisions:')
    pprint(all_precisions)
    print
    print('Recalls:')
    pprint(all_recalls)
    print
    print('F Scores:')
    pprint(all_f_scores)

    print
    print(best.score(test, test_y))




    with open('./models/glm-' + k + '-' + reg + '-' + str(c), 'wb') as outf:
        pickle.dump(best, outf)



if __name__ == "__main__":
    sys.exit(main())









