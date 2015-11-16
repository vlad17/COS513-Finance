"""
Usage: python svm.py train_start, train_end, valid_start, valid_end, test_start, test_end, k, commodity

commodity should be a comma separated list of commodities to study

Loads expanded, summarized feature set of K days, splits data into training and test sets,
and trains using SVM model on the training sets
Selects best parameters based on regularization errors

"""

import numpy as np
import pandas as pd
import glob
import fileinput
from sklearn import svm
from sklearn import decomposition
import pickle
import itertools
import sys
from sklearn.metrics import *
from pprint import pprint


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

    if len(sys.argv) < 8:
        print(__doc__)
        return 1

    (train_start, train_end, valid_start, valid_end, test_start, test_end) = tuple(sys.argv[1:7])

    k = sys.argv[7]
    summarized_dir = '/n/fs/gcf/CORRECT-summary-data-20130401-20151021/' + k + '/'
    summarized_files = glob.glob(summarized_dir + '*.csv')
    print('Reading {} files'.format(len(summarized_files)))

    commodities = ['XAGUSD', 'XAUUSD', 'USDBRL', 'USDCZK', 'USDDKK', 'USDEGP', 'USDHKD', 'USDHUF', 'USDIDR', 'USDILS', 'USDINR', 'USDKRW', 'USDMXN', 'USDMYR', 'USDNOK', 'USDPHP', 'USDPLN', 'USDRUB', 'USDSEK', 'USDSGD', 'USDSKK', 'USDTHB', 'USDTND', 'USDZAR']
    if len(sys.argv) >= 9:
        commodities = sys.argv[8].split(",")

    dates = []
    for sfile in summarized_files:
        date_segment = sfile.split('/')[-1].split('.')[0]
        dates.append(date_segment[:4] + '-' + date_segment[4:6] + '-' + date_segment[6:])

    all_days_array = np.loadtxt(fileinput.input(summarized_files), delimiter = '\t')
    all_days = pd.DataFrame(all_days_array, index=dates)

    # read price differences
    for commodity in commodities:
        price_filename = 'quote-download/' + commodity + '.csv'
        commodity = ".".join([commodity[:3], commodity[3:]])


        price_info, price_changes = get_price_info(price_filename, commodity)
        print ("\n\nCOMMODITY: {}".format(commodity))
        # price_changes = price_changes.shift(-1)

        price_info_slice = price_info[price_info.index.isin(dates)]
        price_changes = price_changes[price_changes.index.isin(dates)]

        all_features = all_days.join(price_info_slice)

        train = all_features[(all_features.index >= train_start) & (all_features.index <= train_end)]
        train_y = price_changes[(price_changes.index >= train_start) & (price_changes.index <= train_end)]

        # PCA
        pca = decomposition.PCA(n_components=20, whiten=True)
        pca.fit(train)
        train = pca.transform(train)

        valid = all_features[(all_features.index >= valid_start) & (all_features.index <= valid_end)]
        valid_y = price_changes[(price_changes.index >= valid_start) & (price_changes.index <= valid_end)]
        valid = pca.transform(valid)

        test = all_features[(all_features.index >= test_start) & (all_features.index <= test_end)]
        test_y = price_changes[(price_changes.index >= test_start) & (price_changes.index <= test_end)]
        test = pca.transform(test)

        print('{} / {} Percentage:{}'.format(sum(np.array(train_y > 0)), len(train_y), sum(np.array(train_y > 0)) * 1.0 / len(train_y)))
        print('{} / {} Percentage:{}'.format(sum(np.array(valid_y > 0)), len(valid_y), sum(np.array(valid_y > 0)) * 1.0 / len(valid_y)))
        print('{} / {} Percentage:{}'.format(sum(np.array(test_y > 0)), len(test_y), sum(np.array(test_y > 0)) * 1.0 / len(test_y)))
        print ()

        use_dual = train.shape[0] < train.shape[1]
        best = None
        best_score = 0
        best_f1_score = 0

        # kernel specific arguments
        kernel_kargs = {'degree': 2}
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            # choose regularization value based on validation error
            for c in itertools.chain(np.logspace(-3.0, 3.0, num=6)):
                model = svm.SVC(C=c, kernel=kernel, **kernel_kargs)
                model.fit(train, train_y)
                valid_score = model.score(valid, valid_y)
                valid_f1_score = f1_score(valid_y, model.predict(valid))
                print ()
                print ("parameters: kernel={} c={}".format(kernel, c))
                print("validation accuracy: " + str(valid_score))
                print('Validation f1 score: ' + str(valid_f1_score))
                if valid_score > best_score:
                    print("---------These params are the best so far--------")
                    best = model
                    best_score = valid_score
        kernel = best.get_params(deep = False)['kernel']
        c = best.get_params(deep = False)['C']

        #train = pd.concat([train, valid])
        #train_y = pd.concat([train_y, valid_y])

        #best.fit(train, train_y)
        print('------------------------')
        print('K: {}'.format(k))
        print('Model (kernel={} C={}) training acc: {} validation acc: {} test accuracy: {}'.format(
            kernel, c, best.score(train, train_y), best.score(valid, valid_y), best.score(test, test_y)))

        test_pred = best.predict(test)
        print ("Predictions on the test set")
        print (test_pred)
        avg_precision = average_precision_score(test_y, test_pred, average='weighted')
        avg_recall = recall_score(test_y, test_pred, average='weighted')
        f_score = f1_score(test_y, test_pred)

        print
        print('Precisions:')
        pprint(avg_precision)
        print
        print('Recalls:')
        pprint(avg_recall)
        print
        print('F Scores:')
        pprint(f_score)

        print
        print('Best accuracy:')
        print(best.score(test, test_y))


        #with open('./models/glm-' + k + '-' + reg + '-' + str(c), 'wb') as outf:
        #    pickle.dump(best, outf)



if __name__ == "__main__":
    sys.exit(main())


