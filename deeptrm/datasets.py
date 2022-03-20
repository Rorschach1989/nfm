import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    """Interface of survival datasets
    """

    @classmethod
    def colon(cls, csv_path):
        """
        The COLON dataset, extracted via the `survival` R Package, refer to the manpage
            https://r-data.pmagunia.com/dataset/r-dataset-package-survival-colon for details
        Read from csv + preprocessing
        2 continuous features + mean-std standardization
        10 categorical features + one-hot encoding (embeddings might be better)
        """
        df = pd.read_csv(csv_path)
        continuous_vars = ['age', 'nodes']
        categorical_vars = ['study', 'rx', 'sex', 'obstruct', 'perfor',
                            'adhere', 'differ', 'extent', 'surg', 'node4']
        df_death = df[df['etype'] == 2]
        prefix = 'feature'
        for v in categorical_vars:
            dummies = pd.get_dummies(df_death[v], prefix=f'{prefix}_{v}')
            df_death = pd.concat([df_death, dummies], axis=1)
        for v in continuous_vars:
            series = df_death[v]
            df_death[f'{prefix}_{v}'] = (series - series.mean()) / (series.std() + 1e-15)
        feature_columns = [c for c in df_death.columns if c.startswith(prefix)]
        feature_df = df_death[feature_columns]
        nan_mask = ~pd.isnull(feature_df).sum(axis=1).values.astype(bool)
        z = df_death[feature_columns].values[nan_mask]
        y = df_death['time'].values[nan_mask]
        delta = df_death['status'].values[nan_mask]
        return cls(y=y, z=z, delta=delta)

    @classmethod
    def whas(cls, dat_path):
        """Whas dataset, see mlr3proba manpage for details https://rdrr.io/cran/mlr3proba/man/whas.html
        Read from dat file (from this ftp directory: ftp://ftp.wiley.com/public/sci_tech_med/survival)
        2 continuous features + mean-std standardization
        3 categorical features + one-hot encoding
        """
        df = pd.read_csv(dat_path, sep=r'\s+', header=None)
        df.columns = ['set', 'case', 't', 'lenfol', 'fstat', 'age', 'sex', 'bmi', 'chf', 'miord', 'nr']
        continuous_vars = ['age', 'bmi']
        categorical_vars = ['sex', 'chf', 'miord']
        prefix = 'feature'
        for v in categorical_vars:
            dummies = pd.get_dummies(df[v], prefix=f'{prefix}_{v}')
            df = pd.concat([df, dummies], axis=1)
        for v in continuous_vars:
            series = df[v]
            df[f'{prefix}_{v}'] = (series - series.mean()) / (series.std() + 1e-15)
        feature_columns = [c for c in df.columns if c.startswith(prefix)]
        feature_df = df[feature_columns]
        nan_mask = ~pd.isnull(feature_df).sum(axis=1).values.astype(bool)
        z = df[feature_columns].values[nan_mask]
        y = df['t'].values[nan_mask]
        delta = df['fstat'].values[nan_mask]
        return cls(y=y, z=z, delta=delta)

    @classmethod
    def metabric(cls, h5_path):
        """Metabric dataset using the dataset available in DeepSurv's directory
        https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/metabric/metabric_IHC4_clinical_train_test.h5

        Requires h5py for preprocessing
        5 continuous features + mean-std standardization
        4 categorical features + one-hot encoding
        """
        f = h5py.File(h5_path, 'r')
        train = f['train']
        test = f['test']
        y = np.concatenate([train['t'][:], test['t'][:]], axis=0)
        delta = np.concatenate([train['e'][:], test['e'][:]], axis=0)
        raw_z = np.concatenate([train['x'][:], test['x'][:]], axis=0)
        _z = []
        for c in (0, 1, 2, 3, 8):  # Standardization
            series = raw_z[:, c]
            _z.append(((series - series.mean()) / (series.std() + 1e-15)).reshape(-1, 1))
        for c in (4, 5, 6, 7):
            _z.append(pd.get_dummies(raw_z[:, c]).values.astype(np.float))
        z = np.concatenate(_z, axis=1)
        return cls(y=y, z=z, delta=delta)

    @classmethod
    def gbsg(cls, h5_path):
        """RotGBSG dataset using the dataset available in DeepSurv's directory
        https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/gbsg/gbsg_cancer_train_test.h5

        Requires h5py for preprocessing
        3 binary features
        4 continuous features + mean-std standardization
        """
        f = h5py.File(h5_path, 'r')
        train = f['train']
        test = f['test']
        y = np.concatenate([train['t'][:], test['t'][:]], axis=0)
        delta = np.concatenate([train['e'][:], test['e'][:]], axis=0)
        z = np.concatenate([train['x'][:], test['x'][:]], axis=0)
        for c in (3, 4, 5, 6):
            series = z[:, c]
            z[:, c] = (series - series.mean()) / (series.std() + 1e-15)
        return cls(y=y, z=z, delta=delta)

    @classmethod
    def support(cls, h5_path):
        """SUPPORT dataset using the the dataset available in DeepSurv's directory
        https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/support/support_train_test.h5

        Use the original + mean-std standardization
        """
        return cls.gbsg(h5_path)

    @classmethod
    def flchain(cls, csv_path):
        """FLchain dataset using the RDataset api
        https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.cs

        using the preprocessing routine of `pycox`
        5 continuous features + mean-std standardization
        3 categorical features + one-hot encoding
        """
        df = pd.read_csv(csv_path)
        # From preprocessing logic of pycox
        df = (df
              .drop(['chapter', 'Unnamed: 0'], axis=1)
              .loc[lambda x: x['creatinine'].isna() == False]
              .reset_index(drop=True)
              .assign(sex=lambda x: (x['sex'] == 'M')))

        categorical = ['sample.yr', 'flc.grp']
        for col in categorical:
            df[col] = df[col].astype('category')
        for col in df.columns.drop(categorical):
            df[col] = df[col].astype('float32')
        continuous_vars = ['age', 'kappa', 'creatinine', 'lambda', 'mgus']
        categorical_vars = ['sex', 'sample.yr', 'flc.grp']
        prefix = 'feature'
        for v in categorical_vars:
            dummies = pd.get_dummies(df[v], prefix=f'{prefix}_{v}')
            df = pd.concat([df, dummies], axis=1)
        for v in continuous_vars:
            series = df[v]
            df[f'{prefix}_{v}'] = (series - series.mean()) / (series.std() + 1e-15)
        feature_columns = [c for c in df.columns if c.startswith(prefix)]
        feature_df = df[feature_columns]
        nan_mask = ~pd.isnull(feature_df).sum(axis=1).values.astype(bool)
        z = df[feature_columns].values[nan_mask]
        y = df['futime'].values[nan_mask]
        delta = df['death'].values[nan_mask]
        return cls(y=y, z=z, delta=delta)

    def __init__(self, y, z, delta, stochastic=True):
        self.sample_size = y.shape[0]
        self.y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        self.delta = torch.tensor(delta, dtype=torch.float).view(-1, 1)
        self.z = torch.tensor(z, dtype=torch.float)
        if stochastic:
            self.y = self.y.clone().detach().requires_grad_(True)

    def __getitem__(self, item):
        return self.z[item], self.y[item], self.delta[item]

    def __len__(self):
        return self.z.shape[0]

    def sort(self):
        order = torch.argsort(self.y, dim=0)[:, 0]
        sort_y = self.y[order]
        sort_delta = self.delta[order]
        sort_z = self.z[order]
        return sort_y, sort_delta, sort_z

    def shuffle(self):
        perm = torch.randperm(self.sample_size)
        self.y = self.y[perm]
        self.z = self.z[perm]
        self.delta = self.delta[perm]

    def cv_split(self, n_folds=5, shuffle=True):
        """Reproduce the splitting CV setup in the paper
        `Deep Extended Hazard Models for Survival Analysis`
        """
        samples_each = self.sample_size // n_folds
        indices = 0
        mask_ = torch.zeros([self.sample_size], dtype=torch.bool)
        train_datasets, valid_datasets, test_datasets = [], [], []
        if shuffle:
            self.shuffle()
        for i in range(n_folds):
            mask = mask_.clone()
            start = indices
            # Last fold might be slightly bigger
            stop = indices + samples_each if i < n_folds - 1 else self.sample_size
            mask[start: stop] = True
            test_z, test_y, test_delta = self[torch.where(mask)[0]]
            train_valid_z, train_valid_y, train_valid_delta = self[torch.where(~mask)[0]]
            n_valid = train_valid_z.shape[0] // 5  # 20% for valid
            valid_z, train_z = train_valid_z[:n_valid], train_valid_z[n_valid:]
            valid_y, train_y = train_valid_y[:n_valid], train_valid_y[n_valid:]
            valid_delta, train_delta = train_valid_delta[:n_valid], train_valid_delta[n_valid:]
            train_datasets.append(self.__class__(y=train_y, z=train_z, delta=train_delta))
            valid_datasets.append(self.__class__(y=valid_y, z=valid_z, delta=valid_delta))
            test_datasets.append(self.__class__(y=test_y, z=test_z, delta=test_delta))
            indices = stop
        return train_datasets, valid_datasets, test_datasets
