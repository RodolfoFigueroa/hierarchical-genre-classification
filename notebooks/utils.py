import pandas as pd
import numpy as np
import torch
from ast import literal_eval

from sklearn.model_selection import train_test_split

import xgboost as xgb


class Welford(object):
    def __init__(self,lst=None):
        self.k = 0
        self.M = 0
        self.S = 0
        
        self.__call__(lst)
    
    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M)*1./self.k
        newS = self.S + (x - self.M)*(x - newM)
        self.M, self.S = newM, newS

    def consume(self,lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)
    
    def __call__(self,x):
        if hasattr(x,"__iter__"):
            self.consume(x)
        else:
            self.update(x)
            
    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std/np.sqrt(self.k)
    
    @property
    def std(self):
        if self.k==1:
            return 0
        return np.sqrt(self.S/(self.k-1))
    
    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


class FileStreamLoader: 
    def __init__(
        self, 
        path, 
        batch_size, 
        buffer_size, 
        feature_names,
        target_name,
        standardize = True,
        means = None,
        stds = None,
        shuffle = False, 
        random_state = 42,
        return_tensors = False
    ):
        assert buffer_size % batch_size == 0, "buffer_size must be a multiple of batch_size"
        assert (not standardize) or (standardize and means and stds), "means and stds can't be empty if standardize=True"

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.feature_names = feature_names
        self.target_name = target_name
        self.total_names = feature_names + [target_name]
        
        self.standardize = standardize
        if standardize:
            self.means = np.array([means[feature] for feature in feature_names])
            self.stds = np.array([stds[feature] for feature in feature_names])
        else:
            self.means = None
            self.stds = None

        self.shuffle = shuffle
        self.rnd = np.random.RandomState(random_state)

        self.ptr = 0
        self.finished = False
        self.file = open(path, "r")
        self.reader = csv.DictReader(self.file)

        self.x_data = None
        self.y_data = None
        self.return_tensors = return_tensors

        self.reload_buffer()

    def reload_buffer(self):
        buffer = []
        self.ptr = 0
        ct = 0
        while ct < self.buffer_size:
            try:
                line = next(self.reader)
            except StopIteration:
                self.finished = True
                break
            to_append = [float(line[feature]) for feature in self.total_names]
            buffer.append(to_append)
            ct += 1

        if self.shuffle == True:
            self.rnd.shuffle(buffer)

        buffer_mat = np.array(buffer)
        n = len(self.feature_names)
        
        if self.standardize:
            x = (buffer_mat[:, :n] - self.means) / self.stds
        else:
            x = buffer_mat[:, :n]
        
        y = buffer_mat[:, n]

        if self.return_tensors:
            self.x_data = torch.FloatTensor(x)
            self.y_data = torch.tensor(y, dtype=torch.long)
        else:
            self.x_data = x
            self.y_data = y

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > len(self.x_data):
            if self.finished:
                self.file.close()
                raise StopIteration
            else:
                self.reload_buffer() 

        start = self.ptr
        end = self.ptr + self.batch_size
        x = self.x_data[start:end, :]
        y = self.y_data[start:end]
        self.ptr += self.batch_size
        return (x, y)


def count_and_filter(l, n=None):
    unique, counts = np.unique(l, return_counts=True)
    if n is None:
        idx = np.argsort(counts)[::-1]
    else:
        idx = np.argsort(counts)[:-n-1:-1]
    return unique[idx], counts[idx]


def _cluster_helper(df):
    df = df.copy()
    df = df[df['genres'].str.len() > 2]
    df['genres'] = df['genres'].apply(literal_eval)
    df = df.explode('genres')
    return df


def cluster_genres(df):
    df = _cluster_helper(df)
    return df.groupby('cluster')['genres'].apply(lambda x: count_and_filter(x))


def genre_clusters(df):
    df = _cluster_helper(df)
    return df.groupby('genres')['cluster'].apply(lambda x: count_and_filter(x)).to_dict()


def get_labels(df, popular=30, rep=8):
    df_sorted = df.sort_values('followers', ascending=False)
    df_popular = df_sorted.head(popular)
    df_rep = df_sorted.groupby('cluster').head(rep)
    df_text = pd.concat([df_popular, df_rep])
    df_text = df_text.drop_duplicates()

    label_text = list(df_text['name'])
    label_coords = df_text[['x', 'y']].to_numpy()
    return label_text, label_coords


def gen_train_test(
    read_path, 
    train_path, 
    train_sample_path, 
    test_path, 
    target, 
    train_size=0.8, 
    random_state=42,
    strategy = 'undersample'
    ):

    with open(read_path, 'r') as f:
        reader = csv.DictReader(f)
        idx, y = [], []
        for i, row in enumerate(reader):
            idx.append(i)
            y.append(row[target])

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, 
        y, 
        stratify = y, 
        train_size = train_size,
        shuffle = True,
        random_state = random_state,
    )
    idx_train_s = set(idx_train)
    idx_train_reshaped = np.array(idx_train).reshape(-1,1)
    
    if strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    idx_train_sampled, _ = sampler.fit_resample(idx_train_reshaped, y_train)
    idx_train_sampled_s = set(list(idx_train_sampled.squeeze()))

    with \
    open(read_path, 'r') as f, \
    open(train_path, 'w') as f_train, \
    open(train_sample_path, 'w') as f_sample, \
    open(test_path, 'w') as f_test:
    
        reader = csv.DictReader(f)
        
        writer_train = csv.DictWriter(f_train, reader.fieldnames)
        writer_train.writeheader()

        writer_test = csv.DictWriter(f_test, reader.fieldnames)
        writer_test.writeheader()

        writer_sample = csv.DictWriter(f_sample, reader.fieldnames)
        writer_sample.writeheader()

        for i, row in enumerate(reader):
            if i in idx_train_s:
                writer_train.writerow(row)
            else:
                writer_test.writerow(row)
            
            if i in idx_train_sampled_s:
                writer_sample.writerow(row)


def flatten_cv(res, vars, y_name, idx=None):
    vars = ['colsample_bytree', 'gamma', 'learning_rate', 'max_depth','min_child_weight', 'subsample']
    if idx is None:
        cv_results = res
    else:
        cv_results = res[res['model'].isin(idx)]
    n_unique = len(cv_results['model'].unique())

    group = cv_results.groupby(vars)

    xp = group['round'].apply(list)
    xp = list([r for r in xp])

    yp = group[y_name].apply(list)
    yp = list([r for r in yp])

    catp = group['model'].apply(list)
    catp = list([r for r in catp])

    minlen = min([len(r) for r in xp])
    xflat = [r for subl in xp for r in subl[:minlen]]
    yflat = [r for subl in yp for r in subl[:minlen]]    
    catflat = [r for subl in catp for r in subl[:minlen]]
    
    dat = np.vstack([xflat, yflat, catflat]).T
    return dat


def xgboost_cv(
    train, 
    fixed_params, 
    param_list,
    num_rounds,
    save_path,
    metric_name,
    early_stopping = 10,
    nfold = 5,
    stratified = True,
    verbose = False
):
    cv_results = pd.DataFrame()
    for i, sample_params in enumerate(param_list):
        params = {**fixed_params, **sample_params}
        cv = xgb.cv(
            params, 
            train,
            num_boost_round = num_rounds,
            early_stopping_rounds = early_stopping,
            nfold = nfold,
            stratified = stratified
        ).copy()

        cv = cv.reset_index()
        cv = cv.rename(columns={'index': 'round'})
        for key, value in sample_params.items():
            cv[key] = value
        cv['model'] = i
        cv_results = pd.concat([cv_results, cv])
        
        train_score = cv[f'train-{metric_name}-mean'].iloc[-1]
        test_score = cv[f'test-{metric_name}-mean'].iloc[-1]
        if verbose:
            print(f"Model {i}. Final train loss: {train_score}. Final test loss: {test_score}")

    cv_results.to_csv(save_path, index=False)


def xgboost_cv_single(
    train,
    params,
    num_rounds,
    metric_name,
    early_stopping = 10,
    nfold = 5,
    stratified = True,
    verbose = False
):
    cv = xgb.cv(
        params, 
        train,
        num_boost_round = num_rounds,
        early_stopping_rounds = early_stopping,
        nfold = nfold,
        stratified = stratified
    )
    train_score = cv[f'train-{metric_name}-mean'].iloc[-1]
    test_score = cv[f'test-{metric_name}-mean'].iloc[-1]

    return test_score


def gen_binary_data(df, target_name, target_value, rng, return_dmatrix=True):
    df_temp = df.copy()
    df_temp['target'] = df_temp[target_name]==target_value

    group = df_temp.groupby('id')
    df_temp = group.first()
    df_temp = df_temp.drop([target_name, 'target'], axis=1)

    idx = group['target'].any()
    idx_pos = idx[idx].index
    idx_neg = idx[~idx].index

    X_pos = df_temp.loc[idx_pos].to_numpy()
    X_neg = df_temp.loc[idx_neg].to_numpy()
    y_pos = np.ones(X_pos.shape[0])
    y_neg = np.zeros(X_neg.shape[0])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg]).reshape(-1, 1)
    
    Xy = np.hstack([X, y])
    rng.shuffle(Xy)

    weight = len(y_neg)/len(y_pos)
    
    if return_dmatrix:
        dtrain = xgb.DMatrix(data=Xy[:, :-1], label=Xy[:, -1])
        return dtrain, weight
    else:
        return Xy, weight