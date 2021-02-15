from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import geopy
import sklearn
from sklearn import metrics
import holidays


## Load data
import json

with open('data/query_prediction/data.json') as json_file:
    data = json.load(json_file)


arrival_df = pd.DataFrame(data)
# Correct timezones
arrival_df['time_plan_ts'] = pd.to_datetime(arrival_df.time_plan).dt.tz_localize(None) + pd.Timedelta(3, 'h')
arrival_df['time_fact_ts'] = pd.to_datetime(arrival_df.time_fact).dt.tz_localize(None) + pd.Timedelta(3, 'h')


# Select only one vehicle and only one route_uuid

route_uuid = arrival_df.route_uuid.unique()[1]
vehicle_uuid = arrival_df.vehicle_uuid.unique()[5]
dates = arrival_df.date.unique()
date_test = '2020-11-22'

df_sample = arrival_df[
    (arrival_df.route_uuid == route_uuid) &
    (arrival_df.vehicle_uuid == vehicle_uuid) 
    & (arrival_df.date.isin(dates))
].sort_values('time_plan')

df_sample = df_sample.iloc[0:].reset_index().drop('index', axis=1)

# Add time, week and holiday columns
labels = [str(i) + '-' + str(j) for (i,j) in zip(np.arange(0, 26, 2), np.arange(0, 26, 2)[1:])]
df_sample['time_bin'] = pd.cut(df_sample.time_plan_ts.dt.hour,
                        bins=np.arange(0, 26, 2),
                        include_lowest=True,
                        labels=labels).astype(str)

df_sample['weekday'] = df_sample.time_plan_ts.apply(lambda x: x.weekday()).astype(str)
holiday_dates = [str(h) for h in holidays.Russia(years=2020)]
df_sample['is_holiday'] = df_sample.date.apply(lambda x: x in holiday_dates).astype(int)
df_sample['is_weekend'] = df_sample.weekday.isin({5,6}).astype(int)

## To timeseries format

# Encode stops
stops = df_sample.apply(lambda x: str(x.latitude) + ' - ' + str(x.longitude), axis=1)

ind2stop = dict(enumerate(stops.unique()))
stop2ind = {s:i for i,s in ind2stop.items()}

df_sample['stop_number'] = stops.apply(lambda x: stop2ind[x])


# Calculate late time
df_sample['plan_time_to_next_stop'] = (df_sample.time_plan_ts.shift(-1) - df_sample.time_plan_ts).apply(lambda x: x.total_seconds())
df_sample.plan_time_to_next_stop = df_sample.plan_time_to_next_stop.fillna(method='ffill')

df_sample['fact_time_to_next_stop'] = (df_sample.time_fact_ts.shift(-1) - df_sample.time_fact_ts).apply(lambda x: x.total_seconds())
df_sample.fact_time_to_next_stop = df_sample.fact_time_to_next_stop.fillna(df_sample.plan_time_to_next_stop)
df_sample['time_off'] = df_sample.fact_time_to_next_stop - df_sample.plan_time_to_next_stop

print(df_sample.head())


# Encode string columns as numbers

from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(dataframe, col):
    df = dataframe.copy()
    oh = OneHotEncoder()
    array = oh.fit_transform(df[col].values.reshape((-1, 1))).toarray()
    cols = [col + '_' + str(i) for i in range(array.shape[1])]
    df[cols] = array
    return df.drop(col, axis=1)

encoded = df_sample[['time_off', 'plan_time_to_next_stop', 'fact_time_to_next_stop', 'time_bin', 'stop_number', 'weekday', 'is_weekend', 'is_holiday', 'date']].iloc[:-1]

encoded = one_hot_encode(encoded, 'date')
encoded = one_hot_encode(encoded, 'time_bin')
encoded = one_hot_encode(encoded, 'stop_number')
encoded = one_hot_encode(encoded, 'weekday')


## Add information about recent stops

def df_to_supervized(series, columns, num_recent, fillna_method='mean'):
    df = series.copy()
    for col, num in zip(columns, num_recent):
        if sum(df[col].isna() > 0):
            indicator = col + '_isna'
            df[indicator] = df[col].isna().astype(int)
            if fillna_method == 'mean':
                method = df[col].mean()
            else:
                method = fillna_method
            df[col] = df[col].fillna(method)

        for i in range(num):
            col_name = col + '_-' + str(i+1)
            df[col_name] = df[col].shift(i+1)
    
    return df.iloc[max(num_recent):]



# Drop peaks 

peaks_index = encoded.loc[encoded.plan_time_to_next_stop > 10 * 60].index

encoded = encoded.drop(peaks_index)

cols = encoded.columns
nums = [0, 8, 5]

supervized = df_to_supervized(encoded, cols, nums, fillna_method='mean')


# Divide into train and test

test = supervized[(supervized.date_0 == 1) | (supervized.date_1 == 1)]
test1 = supervized[(supervized.date_0 == 1)]
test2 = supervized[(supervized.date_1 == 1)]
train = supervized[(supervized.date_0 == 0) & (supervized.date_1 == 0)]


y_train = train['time_off']
X_train = train.drop(['fact_time_to_next_stop', 'time_off'], axis=1)

y_test = test['time_off']
X_test = test.drop(['fact_time_to_next_stop', 'time_off'], axis=1)


# Functions for model validation
from sklearn.metrics import mean_squared_error, mean_absolute_error

def validate_predictions(_y_true, _y_pred, _plan_time=None, scale=1./60, plot_graph=True, plan_thresh=9, late_thresh=[-3,4]):
    y_true = _y_true.values * scale
    y_pred = _y_pred * scale
    err = y_true - y_pred
    print('MSE: ', mean_squared_error(y_pred=y_pred, y_true=y_true))
    print('MAE: ', mean_absolute_error(y_pred=y_pred, y_true=y_true))

    if _plan_time is not None:
        plan_time = _plan_time * scale
    if plan_time is not None:
        print('MSE when plan_time < {}m: '.format(plan_thresh), 
                mean_squared_error(y_pred=y_pred[plan_time < plan_thresh], y_true=y_true[plan_time < plan_thresh]))
        print('MAE when plan_time < {}m: '.format(plan_thresh),
                mean_absolute_error(y_pred=y_pred[plan_time < plan_thresh], y_true=y_true[plan_time < plan_thresh]))

    # print(err[(err > late_thresh[1]) | (err < late_thresh[0])])
    print('num errors > {}m: {} ({}%)'.format(late_thresh, np.sum((err > late_thresh[1]) | (err < late_thresh[0])), 
                                                    round(np.mean((err > late_thresh[1]) | (err < late_thresh[0])) * 100, 1)))
    if plan_time is not None:
        print('num errors > {}m when plan time < {}m: {} ({}%)'.format(late_thresh, plan_thresh,
             np.sum((err[plan_time < plan_thresh] > late_thresh[1]) | (err[plan_time < plan_thresh] < late_thresh[0])), 
             round(np.mean((err[plan_time < plan_thresh] > late_thresh[1]) | (err[plan_time < plan_thresh] < late_thresh[0])) * 100, 1)))

    if plot_graph:
        peaks = []
        if plan_time is not None:
            peaks = np.where(plan_time >= plan_thresh)[0]

        plt.figure(figsize=(16,8))
        plt.plot(y_true)
        plt.plot(y_pred)
        plt.legend(['ground truth', 'predictions'])
        plt.show()

        plt.figure(figsize=(16,8))
        plt.title('Error')
        plt.plot(y_true - y_pred)
        plt.plot([late_thresh[0]] * len(y_true), c='pink')
        plt.plot([late_thresh[1]] * len(y_true), c='pink')
        plt.vlines(peaks, late_thresh[0], late_thresh[1], colors='grey')
        plt.show()


# Define model
from xgboost import XGBRegressor

best_params = {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 1000, 'reg_alpha': 0.1, 'reg_lambda': 0.01, 'subsample': 0.6}

model = XGBRegressor(**best_params)
                    
model.fit(X_train, y_train)


## Validate model

# On train

# estimate by plan time
validate_predictions(y_train, np.zeros_like(y_train), _plan_time=X_train.plan_time_to_next_stop)

# estimate by model predictions
validate_predictions(y_train, model.predict(X_train), _plan_time=X_train.plan_time_to_next_stop)


t1 = 255
## Test day1 

# estimate by plan time
validate_predictions(y_test[:t1], np.zeros(len(y_test[:t1])), _plan_time=X_test.plan_time_to_next_stop[:t1])

# estimate by model predictions
validate_predictions(y_test[:t1], model.predict(X_test[:t1]), _plan_time=X_test.plan_time_to_next_stop[:t1])


# ### Test day 2

# estimate by plan time
validate_predictions(y_test[t1:], np.zeros(len(y_test))[t1:], _plan_time=X_test.plan_time_to_next_stop[t1:])

# estimate by model predctions
validate_predictions(y_test[t1:], model.predict(X_test[t1:]), _plan_time=X_test.plan_time_to_next_stop[t1:])

