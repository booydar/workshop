import numpy as np
import pandas as pd
from datetime import timedelta
import time
import random
from functools import wraps
from time import time

from sklearn.preprocessing import LabelEncoder


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end - start))
        return result

    return wrapper


class FeatureGenerator(object):
    def __init__(self, n=1, insert_blank=0.1):
        self.feature_types_list = ['labels', 'continous', 'mixed', 'ohe']
        self.label_type_list = ['numerical', 'string', 'mixed']
        self.distribution_list = ['normal', 'uniform']
        self.n = n
        self.insert_blank = insert_blank

    def label_features(self, label_type='mixed', nb_unique_labels=10):
        """
        Функция генерирует лейблы в форматах str или int
        :param label_type: тип лейбла: 'string' - str only; numeical - 'int' only; 'mixed' - mix of 'str' and 'int'
        :param nb_unique_labels: max nb of unique labels of *every* type ('mixed' can return up to 2*nb_unique_labels)
        :return: feature vector of shape (n,)
        """
        assert label_type in self.label_type_list
        if label_type == 'mixed' or label_type == 'string':
            unique_str_labels = ['label_' + str(i) for i in range(nb_unique_labels)]
            samples = np.random.choice(unique_str_labels, size=(self.n,), replace=True)
        if label_type == 'mixed' or label_type == 'numerical':
            unique_num_labels = [i for i in range(nb_unique_labels)]
            samples = np.append(samples, np.random.choice(unique_num_labels, size=(self.n,), replace=True))

        if label_type == 'mixed':
            return np.random.choice(samples, size=(self.n,), replace=False)
        else:
            return samples

    def continous_features(self, distribution='normal', **kwargs):
        """
        :param distribution: 'normal' or 'uniform'
        :param kwargs: 'low' and 'high' for 'uniform'; 'loc' and 'scale' for 'normal'
        :return: feature vector of shape (n,)
        """
        assert distribution in self.distribution_list
        if distribution == 'normal':
            if 'loc' in kwargs:
                loc = kwargs['loc']
            else:
                loc = np.random.uniform(0, 10)

            if 'scale' in kwargs:
                scale = kwargs['scale']
            else:
                scale = np.random.uniform(1, 10)
            return np.random.normal(loc, scale, size=(self.n,))
        else:
            if 'low' in kwargs:
                low = kwargs['low']
            else:
                low = np.random.uniform(-1, 1)

            if 'high' in kwargs:
                high = kwargs['high']
            else:
                high = np.random.uniform(2, 10000)
            return np.random.uniform(low, high, size=(self.n,))

    def OHE_features(self, n_features=None):
        """
        :param n_features: int
        :return: shuffle identity matrix of shape(n,n_features)
        """
        if not n_features:
            n_features = np.random.randint(1, 20)
        matrix_feature = np.zeros((self.n, n_features), dtype='int8')
        for i in range(matrix_feature.shape[0]):
            matrix_feature[i, random.choice(list(np.arange(0, n_features)))] = 1
        return matrix_feature

    @staticmethod
    def make_target(features_df, feature_types_dict, dependency):
        target = np.zeros(len(features_df))
        for feature in features_df.columns.values:
            if feature in feature_types_dict:
                feature_type = feature_types_dict[feature]
                if feature_type == 'labels':
                    feature_vals = LabelEncoder().fit_transform(features_df[feature].values)
                    target += feature_vals.dot(np.random.normal(0, 1))
                elif feature_type == 'ohe':
                    feature_vals = features_df[feature].values
                    weight = np.random.normal(0, 1, size=features_df.shape[0])
                    target += feature_vals.dot(weight)
                else:
                    feature_vals = features_df[feature].values
                    target += feature_vals * np.random.randint(1, 1000) / 1000
            # print(target)
            target = (target - np.mean(target)) / np.std(target)
        if dependency == 'classification':
            n_classes = np.random.randint(4, 12)
            bins = np.histogram(target, n_classes)[1]
            features_df['target'] = np.digitize(target, bins)
        else:
            features_df['target'] = target

        return features_df

    def make_feature_dataset(self, n_features=10, feature_types='mixed',
                             dependency=None,
                             **kwargs):
        """

        :param n_features: nb of features to generate
        :param feature_types: one of 'labels', 'continous', 'mixed'
        :param dependency: <>
        :return: pd.DataFrame({ feature_1:np.array(n_values,)
                                ...
                                feature_n_features:np.array(n_values,)})
        """
        assert feature_types in self.feature_types_list
        if not n_features > 0 and type(n_features) == int:
            raise AssertionError('n_features must be of type <int> and in range(1,1000)')
        features_names = ['feature_' + str(i) for i in range(n_features)]
        features_df = pd.DataFrame(columns=features_names)
        feature_makers = [(self.label_features, 'labels'),
                          (self.continous_features, 'continous'),
                          (self.OHE_features, 'ohe')]
        feature_types_dict = dict()
        for feature in features_names:
            if feature_types == 'labels':
                features_df[feature] = self.label_features()
                feature_types_dict[feature] = 'labels'
            if feature_types == 'continous':
                features_df[feature] = self.continous_features()
                feature_types_dict[feature] = 'continous'
            if feature_types == 'ohe':
                temp_ohe_vals = self.OHE_features()
                for i, val in enumerate(temp_ohe_vals.T):
                    features_df[feature + "_ohe_" + str(i)] = val
                    feature_types_dict[feature + "_ohe_" + str(i)] = 'ohe'
            if feature_types == 'mixed':
                feature_maker = random.choice(feature_makers)
                if feature_maker[1] == 'ohe':
                    print(1)
                    temp_ohe_vals = feature_maker[0]()
                    for i, val in enumerate(temp_ohe_vals.T):
                        features_df[feature + "_ohe_" + str(i)] = val
                        feature_types_dict[feature + "_ohe_" + str(i)] = 'ohe'
                    feature_makers.pop()
                else:
                    features_df[feature], feature_types_dict[feature] = feature_maker[0](), feature_maker[1]

        if not dependency:
            dependency = random.choice(['classification', 'regression'])
        features_df = self.make_target(features_df, dependency=dependency, feature_types_dict=feature_types_dict)
        features_df = features_df.dropna(axis='columns')
        return features_df


class SimpleProcessMaker(object):
    def __init__(self,
                 probas,
                 stages,
                 dt_start=pd.datetime(year=2020, month=1, day=1),
                 p_keep_order=0.85):
        self.probas = probas
        self.stages = stages
        self.dt_start = dt_start
        self.p_keep_opder = p_keep_order

    @staticmethod
    def make_proba(p_min=1, p_max=100, runs=2):
        return np.mean(np.random.randint(p_min, p_max, runs)) / 100

    @staticmethod
    def timestamps_generator(n):
        res = []
        for i in range(n):
            if i == 0:
                res.append(timedelta(days=i))
            else:
                res.append(res[i-1] + timedelta(days=np.random.randint(1, 10)))
        return res

    def make_process_sample(self, idx):
        p = np.array([self.make_proba() for _ in range(len(self.probas))])
        proc_sample = p > self.probas
        proc_sample = self.stages[proc_sample]

        for i in range(len(proc_sample) - 1):
            swap_proba = np.random.normal(0, 1)
            if swap_proba > self.p_keep_opder:
                proc_sample[i], proc_sample[i + 1] = proc_sample[i + 1], proc_sample[i]

        date_increment = self.timestamps_generator(len(proc_sample))

        dates_samples = [self.dt_start + date_increment[i] for i in range(len(proc_sample))]

        ids = [idx for _ in range(len(proc_sample))]

        return pd.DataFrame({'id': ids,
                             'stages': proc_sample,
                             'dt': dates_samples})
