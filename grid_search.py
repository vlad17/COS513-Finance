"""
Usage: python grid_search.py result_filename

result_filename is the name of the file

Loop over all possible commodities/indices, all possible values of k, all possible models and all possible model parameters
to determine the values that have the most improvement over the baseline accuracy.
"""

import csv
import math
import numpy as np
import pandas as pd
import glob
import fileinput
from sklearn import svm
from sklearn import decomposition
from sklearn.linear_model import LinearRegression, LogisticRegression

import pickle
import itertools
import sys
from sklearn.metrics import *
from pprint import pprint


# Dictionary that maps from financial index to (column name, whether to compute the log return). If True, it means
# we should regress against the log return.
_INDICES = {
    'XAGUSD': ('XAG.USD', True),
    'XAUUSD': ('XAU.USD', True),
    'USDBRL': ('USD.BRL', True),
    'USDCZK': ('USD.CZK', True),
    'USDDKK': ('USD.DKK', True),
    'USDEGP': ('USD.EGP', True),
    'USDHKD': ('USD.HKD', True),
    'USDHUF': ('USD.HUF', True),
    'USDIDR': ('USD.IDR', True),
    'USDILS': ('USD.ILS', True),
    'USDINR': ('USD.INR', True),
    'USDKRW': ('USD.KRW', True),
    'USDMXN': ('USD.MXN', True),
    'USDMYR': ('USD.MYR', True),
    'USDNOK': ('USD.NOK', True),
    'USDPHP': ('USD.PHP', True),
    'USDPLN': ('USD.PLN', True),
    'USDRUB': ('USD.RUB', True),
    'USDSEK': ('USD.SEK', True),
    'USDSGD': ('USD.SGD', True),
    'USDSKK': ('USD.SKK', True),
    'USDTHB': ('USD.THB', True),
    'USDTND': ('USD.TND', True),
    'USDZAR': ('USD.ZAR', True),
    'CJ': ('CJ.Adjusted', True),
    'CY': ('CY.Adjusted', True),
    'CZ': ('CZ.Adjusted', True),
    'D7L': ('D7L.Adjusted', True),
    'DAC': ('DAC.Adjusted', True),
    'DBL': ('DBL.Adjusted', True),
    'DEN': ('DEN.Adjusted', True),
    'DX': ('DX.Adjusted', True),
    'EFM': ('EFM.Adjusted', True),
    'EJ': ('EJ.Adjusted', True),
    'ESK': ('ESK.Adjusted', True),
    'EVC': ('EVC.Adjusted', True),
    'EWN': ('EWN.Adjusted', True),
    'EXR': ('EXR.Adjusted', True),
    'FI': ('FI.Adjusted', True),
    'GCU': ('GCU.Adjusted', True),
    'GD': ('GD.Adjusted', True),
    'GL': ('GL.Adjusted', True),
    'GOC': ('GOC.Adjusted', True),
    'GX': ('GX.Adjusted', True),
    'HE': ('HE.Adjusted', True),
    'HMO': ('HMO.Adjusted', True),
    'HP': ('HP.Adjusted', True),
    'IN': ('IN.Adjusted', True),
    'IR': ('IR.Adjusted', True),
    'IS': ('IS.Adjusted', True),
    'KA': ('KA.Adjusted', True),
    'KEO': ('KEO.Adjusted', True),
    'KEP': ('KEP.Adjusted', True),
    'KGB': ('KGB.Adjusted', True),
    'KMF': ('KMF.Adjusted', True),
    'KOL': ('KOL.Adjusted', True),
    'KRA': ('KRA.Adjusted', True),
    'KRK': ('KRK.Adjusted', True),
    'KRU': ('KRU.Adjusted', True),
    'KT': ('KT.Adjusted', True),
    'LPE': ('LPE.Adjusted', True),
    'LPS': ('LPS.Adjusted', True),
    'M6J': ('M6J.Adjusted', True),
    'MAE': ('MAE.Adjusted', True),
    'MAS': ('MAS.Adjusted', True),
    'MBE': ('MBE.Adjusted', True),
    'MCC': ('MCC.Adjusted', True),
    'MCD': ('MCD.Adjusted', True),
    'MDB': ('MDB.Adjusted', True),
    'MFB': ('MFB.Adjusted', True),
    'MKC': ('MKC.Adjusted', True),
    'MM': ('MM.Adjusted', True),
    'MNC': ('MNC.Adjusted', True),
    'MXB': ('MXB.Adjusted', True),
    'NI': ('NI.Adjusted', True),
    'NL': ('NL.Adjusted', True),
    'NP': ('NP.Adjusted', True),
    'NS': ('NS.Adjusted', True),
    'OIL': ('OIL.Adjusted', True), 
    'OMM': ('OMM.Adjusted', True),
    'PGP': ('PGP.Adjusted', True),
    'PH': ('PH.Adjusted', True),
    'PLE': ('PLE.Adjusted', True),
    'PZ': ('PZ.Adjusted', True),
    'QI': ('QI.Adjusted', True),
    'RB': ('RB.Adjusted', True),
    'RG': ('RG.Adjusted', True),
    'RL': ('RL.Adjusted', True),
    'RM': ('RM.Adjusted', True),
    'RMB': ('RMB.Adjusted', True),
    'RT': ('RT.Adjusted', True),
    'SD': ('SD.Adjusted', True),
    'SF': ('SF.Adjusted', True),
    'SI': ('SI.Adjusted', True),
    'SIL': ('SIL.Adjusted', True),
    'SIR': ('SIR.Adjusted', True),
    'SMC': ('SMC.Adjusted', True),
    'SP': ('SP.Adjusted', True),
    'SPY': ('SPY.Adjusted', True),
    'SS': ('SS.Adjusted', True),
    'STI': ('STI.Adjusted', True),
    'STR': ('STR.Adjusted', True),
    'STZ': ('STZ.Adjusted', True),
    'TC': ('TC.Adjusted', True),
    'TR': ('TR.Adjusted', True),
    'TT': ('TT.Adjusted', True),
    'TZ6': ('TZ6.Adjusted', True),
    'UA': ('UA.Adjusted', True),
    'UCM': ('UCM.Adjusted', True),
    'UN': ('UN.Adjusted', True),
    'VIX': ('VIX.Adjusted', True),
    'ZN': ('ZN.Adjusted', True),
}

_INDICES = {
    'VIX': ('VIX.Adjusted', True),
    'aabear': ('Mid Price', False),
    'aabull': ('Mid Price', False),
    'aaneut': ('Mid Price', False),
    'gsci': ('Mid Price', True),
    'US10': ('Mid Price', True),  
}
 

# _K_VALUES = [30, 100, 300, 1000]
_K_VALUES = [30, 100, 300]

# False means do not perform PCA
_N_COMPONENTS = [10, 30, 100, 300, False]

# Scale log returns by this number
_SCALE = 1000000

# Train : validation : test ratio (out of 100)
_SPLIT_RATIOS = (60.0, 20.0, 20.0)
_SPLIT_TOTAL = sum(_SPLIT_RATIOS)

# Name of the CSV file to save as
_RESULT_FILENAME = "grid_search_result.csv"
if len(sys.argv) >= 2:
    _RESULT_FILENAME = sys.argv[1]


def get_price_info(price_filename, column, change_in_price=False):
    """
    Get prices of an index. If change_in_price is True, compute the log return in the price.
    Return a tuple (price_info, time_series) where price_info contains the price information and
    the moving average; time_series contains either the price or the log return in the price.
    """
    prices = []
    prices = pd.read_csv(price_filename, sep=' ', index_col=0)
    five_day_avg = pd.Series(pd.rolling_mean(prices[column], 5), name='five_day_avg')
    ten_day_avg = pd.Series(pd.rolling_mean(prices[column], 10), name='ten_day_avg')
    thirty_day_avg = pd.Series(pd.rolling_mean(prices[column], 30), name='thirty_day_avg')

    price_info = pd.DataFrame(pd.concat([prices[column], five_day_avg, ten_day_avg, thirty_day_avg], axis=1))
    if not change_in_price:
        return price_info, prices[column]
    else:
        price_changes_series = pd.Series([_SCALE * math.log(x) for x in prices[column]], dtype=int, index=price_info.index.values)
        price_changes_series = price_changes_series.diff()
        return price_info, price_changes_series


def main():

    # Dictionary that maps (index, k, n) to (train_R^2, validation_R^2, test_R^2, baseline_train_R^2, baseline_validation_R^2, baseline_test_R^2, improvement)
    _R_SQUARED_VALUES = {}

    for k in _K_VALUES:
        # Read the summary data files
        summarized_dir = '/n/fs/gcf/CORRECT-summary-data-20130401-20151021/{}/'.format(k)
        summarized_files = glob.glob(summarized_dir + '*.csv')
        print('\n\n------------- k = {} ------------'.format(k))
        print('Reading {} files'.format(len(summarized_files)))

        dates = []
        for sfile in summarized_files:
            date_segment = sfile.split('/')[-1].split('.')[0]
            dates.append(date_segment[:4] + '-' + date_segment[4:6] + '-' + date_segment[6:])

        all_days_array = np.loadtxt(fileinput.input(summarized_files), delimiter = '\t')
        all_days = pd.DataFrame(all_days_array, index=dates)

        for index in _INDICES:
            # Read price information for the index
            (column_name, change_in_price) = _INDICES[index]
            price_filename = 'quote-download/' + index + '.csv'

            print ("\n\nProcessing the index: {}\n".format(index))
            price_info, index_time_series = get_price_info(price_filename, column_name, change_in_price)

            # Filter the price info by date and by NaN
            price_info_slice = price_info[price_info.index.isin(dates)]
            index_time_series = index_time_series[index_time_series.index.isin(dates)]
            index_time_series = index_time_series[np.isfinite(index_time_series)]

            # Calculate the R^2 values achieved by the baseline without using GDELT data
            all_features = price_info_slice
            all_features = all_features[np.isfinite(all_features['thirty_day_avg'])]
            all_features = all_features.sort_index()
            index_time_series = index_time_series.sort_index()

            # Obtain the dates for training, validation, and testing
            num_dates = len(all_features)
            train_end = int(_SPLIT_RATIOS[0] / _SPLIT_TOTAL * num_dates)
            valid_end = int((_SPLIT_RATIOS[0] + _SPLIT_RATIOS[1]) / _SPLIT_TOTAL * num_dates)
            train_dates = all_features.index[:train_end]
            valid_dates = all_features.index[train_end:valid_end]
            test_dates = all_features.index[valid_end:]

            # Partition into training, validation and test set
            train = all_features[(all_features.index >= train_dates[0]) & (all_features.index <= train_dates[-1])]
            train_y = index_time_series[(all_features.index >= train_dates[0]) & (all_features.index <= train_dates[-1])]

            valid = all_features[(all_features.index >= valid_dates[0]) & (all_features.index <= valid_dates[-1])]
            valid_y = index_time_series[(all_features.index >= valid_dates[0]) & (all_features.index <= valid_dates[-1])]

            test = all_features[(all_features.index >= test_dates[0]) & (all_features.index <= test_dates[-1])]
            test_y = index_time_series[(all_features.index >= test_dates[0]) & (all_features.index <= test_dates[-1])]

            # Perform linear regression
            model = LinearRegression(normalize=True, n_jobs=-1)
            model.fit(train, train_y)
            baseline_train_score = model.score(train, train_y)
            baseline_valid_score = model.score(valid, valid_y)
            baseline_test_score = model.score(test, test_y)

            # Now actually calculate the R^2 values with GDELT data 
            all_features = all_days.join(price_info_slice)
            all_features = all_features[np.isfinite(all_features['thirty_day_avg'])]
            all_features = all_features.sort_index()
            index_time_series = index_time_series.sort_index()

            # Obtain the dates for training, validation, and testing
            train_dates = all_features.index[:train_end]
            valid_dates = all_features.index[train_end:valid_end]
            test_dates = all_features.index[valid_end:]

            for n in _N_COMPONENTS:
                print ("Number of PCA components: {}".format(n))
                # Partition into training, validation and test set
                train = all_features[(all_features.index >= train_dates[0]) & (all_features.index <= train_dates[-1])]
                train_y = index_time_series[(all_features.index >= train_dates[0]) & (all_features.index <= train_dates[-1])]

                valid = all_features[(all_features.index >= valid_dates[0]) & (all_features.index <= valid_dates[-1])]
                valid_y = index_time_series[(all_features.index >= valid_dates[0]) & (all_features.index <= valid_dates[-1])]

                test = all_features[(all_features.index >= test_dates[0]) & (all_features.index <= test_dates[-1])]
                test_y = index_time_series[(all_features.index >= test_dates[0]) & (all_features.index <= test_dates[-1])]

                # PCA
                if n and n < len(all_features.columns):
                    pca = decomposition.PCA(n_components=n)
                    pca.fit(train)
                    train = pca.transform(train)
                    valid = pca.transform(valid)
                    test = pca.transform(test)

                # Perform linear regression
                model = LinearRegression(normalize=True, n_jobs=-1)
                model.fit(train, train_y)
                train_score = model.score(train, train_y)
                valid_score = model.score(valid, valid_y)
                test_score = model.score(test, test_y)

                print ("Train R^2: {}".format(train_score))
                print ("Valid R^2: {}".format(valid_score))
                print ("Test R^2: {}".format(test_score))

                improvement = valid_score - baseline_valid_score
                print ("Improvement over baseline: {}".format(improvement))
                print ()

                _R_SQUARED_VALUES[(index, k, n, num_dates)] = (train_score, valid_score, test_score, baseline_train_score, baseline_valid_score, baseline_test_score, improvement)

    _R_SQUARED_VALUES = list(_R_SQUARED_VALUES.items())
    # Sort by the R^2 value on validation set
    _R_SQUARED_VALUES.sort(key = lambda x: x[1][1], reverse=True)
    print (_R_SQUARED_VALUES)

    # Save the results in a CSV file
    with open(_RESULT_FILENAME, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in _R_SQUARED_VALUES:
            row = list(row[0]) + list(row[1])
            writer.writerow(row)
    print ("Wrote to file {}".format(_RESULT_FILENAME))


if __name__ == "__main__":
    sys.exit(main())

