import numpy as np
import pandas as pd
import glob
import fileinput
from sklearn.linear_model import LogisticRegression
import pickle


summarized_dir = '../summarized_data/'
summarized_files = glob.glob(summarized_dir + '*.csv')

train_start = '2013-04-01'
train_end = '2013-07-31'
test_start = '2013-08-01'
test_end = '2013-09-30'

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

price_info, price_changes = get_price_info(price_filename, commodity)
price_info_slice = price_info[price_info.index.isin(dates)]
price_changes = price_changes[price_changes.index.isin(dates)]

all_features = all_days.join(price_info_slice)

train = all_features[(all_features.index >= train_start) & (all_features.index <= train_end)]
train_y = price_changes[(price_changes.index >= train_start) & (price_changes.index <= train_end)]

test = all_features[(all_features.index >= test_start) & (all_features.index <= test_end)]
test_y = price_changes[(price_changes.index >= test_start) & (price_changes.index <= test_end)]


# logistic regression
model1 = LogisticRegression()

model1.fit(train, train_y)
print 'Model error: {}'.format(model1.score(test, test_y))


with open('./models/glm', 'w') as outf:
    pickle.dump(model1, outf)










