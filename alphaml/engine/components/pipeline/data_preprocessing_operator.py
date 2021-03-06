import typing
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, \
    MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, DATA_PERPROCESSING


class ImputerOperator(Operator):
    # TODO: Different inpute strategy
    def __init__(self, label_col=-1, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_imputer', params)
        self.label_col = label_col

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a ImputeOperator is a pd.Dataframe
        assert len(dm_list) == 1 and isinstance(dm_list[0], pd.DataFrame)
        self.check_phase(phase)

        input_df = dm_list[0]
        df = self.impute_df(input_df)
        dm = DataManager()

        label_col = df.columns[self.label_col] if phase == 'train' else None
        dm.set_col_type(df, label_col)
        data = df.values
        if phase == 'train':
            # Swap label index to -1
            swap_list=list(range(data.shape[1]))
            del(swap_list[self.label_col])
            swap_list.append(self.label_col)
            data=data[:,swap_list]
            dm.train_X = data[:, :-1]
            dm.train_y = data[:, -1]
        else:
            dm.test_X = data
        return dm

    def impute_categorical(self, col):
        return col.fillna(col.mode().iloc[0])

    def impute_float(self, col) -> pd.Series:
        mean_val = col.mean()
        return col.fillna(mean_val)

    def impute_discrete(self, col) -> pd.Series:
        mean_val = int(col.mean())
        return col.fillna(mean_val)

    def impute_col(self, col, datatype) -> pd.Series:
        if datatype == "categorical":
            return self.impute_categorical(col)
        elif datatype == "float":
            return self.impute_float(col)
        elif datatype == "discrete":
            return self.impute_discrete(col)
        else:
            raise TypeError("Required datatype to be categorical, float or discrete")

    def impute_df(self, df) -> pd.DataFrame:
        for col in list(df.columns):
            dtype = df[col].dtype
            if dtype in [np.int, np.int16, np.int32, np.int64]:
                df[col] = self.impute_col(df[col], "discrete")
            elif dtype in [np.float, np.float16, np.float32, np.float64, np.float128, np.double]:
                df[col] = self.impute_col(df[col], "float")
            elif dtype in [np.str, np.str_, np.string_, np.object]:
                df[col] = self.impute_col(df[col], "categorical")
            else:
                raise TypeError("Unknown data type:", dtype)
            # print("Coltype:", dtype)
        return df


class LabelEncoderOperator(Operator):
    def __init__(self, label_col=-1, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_labelencoder', params)
        self.label_col = label_col
        self.label_encoder = LabelEncoder()

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a LabelEncoderOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        # Use LabelEncoder to transform categorical labels into numerical values
        if phase == 'train':
            dm.train_y = self.label_encoder.fit_transform(dm.train_y)
        return dm


class FeatureEncoderOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for OneHotEncoder, 1 for OrdinalEncoder
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_onehotencoder', params)
            self.encoder = OneHotEncoder(handle_unknown="ignore")
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_ordinalencoder', params)
            self.encoder = OrdinalEncoder()
        else:
            raise ValueError("Invalid params in FeatureEncoderOperator. Expected {0,1}")

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a FeatureEncoderOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types

        # Encode categorical features
        categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]
        other_index = [i for i in range(len(feature_types)) if feature_types[i] != "Categorical"]

        # Check if there are no categorical features in train_x
        if len(categorical_index) == 0:
            return dm

        if phase == 'train':
            x = dm.train_X
        else:
            x = dm.test_X

        # Transform categorical features via *Encoder
        if self.params == 0:  # One-hot
            categorical_x = x[:, categorical_index]
            other_x = x[:, other_index]

            if phase == 'train':
                categorical_x = self.encoder.fit_transform(categorical_x).toarray()
            else:
                categorical_x = self.encoder.transform(categorical_x).toarray()
            categorical_features = ["One-Hot"] * categorical_x.shape[1]
            other_features = [feature_types[i] for i in other_index]

            x = np.hstack((categorical_x, other_x)).astype(np.float)
            dm.feature_types = np.concatenate((categorical_features, other_features))
        elif self.params == 1:  # Ordinal
            if phase == 'train':
                x[:, categorical_index] = self.encoder.fit_transform(x[:, categorical_index])
            else:
                x[:, categorical_index] = self.encoder.transform(x[:, categorical_index])
            x = x.astype(np.float)

            for index in categorical_index:
                dm.feature_types[index] = 'Discrete'

        if phase == 'train':
            dm.train_X = x
        else:
            dm.test_X = x

        return dm


class ScalerOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for StandardScaler, 1 for MinMaxScaler, 2 for MaxAbsScaler
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_standardscaler', params)
            self.scaler = StandardScaler()
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_minmaxscaler', params)
            self.scaler = MinMaxScaler()
        elif params == 2:
            super().__init__(DATA_PERPROCESSING, 'dp_maxabsscaler', params)
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Invalid params for ScalerOperator. Expected {0,1,2}")

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a ScalerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        # Check if there are no numerical features in train_x
        if len(numercial_index) == 0:
            return dm

        if phase == 'train':
            x = dm.train_X
            x[:, numercial_index] = self.scaler.fit_transform(x[:, numercial_index])
            dm.train_X = x
        else:
            x = dm.test_X
            x[:, numercial_index] = self.scaler.transform(x[:, numercial_index])
            dm.test_X = x
        return dm


class NormalizerOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for l2 norm, 1 for l1 norm
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_l2normalizer', params)
            self.normalizer = Normalizer(norm='l2')
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_l1normalizer', params)
            self.normalizer = Normalizer(norm='l1')
        else:
            raise ValueError("Invalid params for NormalizerOperator. Expected {0,1}")

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a NormalizerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numericial_index = [i for i in range(len(feature_types))
                            if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        # Check if there are no numerical features in train_x
        if len(numericial_index) == 0:
            return dm

        if phase == 'train':
            x = dm.train_X
            x[:, numericial_index] = self.normalizer.fit_transform(x[:, numericial_index])
            dm.train_X = x
        else:
            x = dm.test_X
            x[:, numericial_index] = self.normalizer.transform(x[:, numericial_index])
            dm.test_X = x
        return dm


# TODO: Bucketizer

class ConstantRemoverOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_constant_remover', params)
        self.constant_indices = []

    def operate(self, dm_list: typing.List, phase='train'):
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numericial_index = [i for i in range(len(feature_types))
                            if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        if phase == 'train':
            x = dm.train_X
            for index in numericial_index:
                feature = x[:, index]
                if len(set(feature)) == 1:  # Constant feature
                    self.constant_indices.append(index)
            self.constant_indices.reverse()
            for index in self.constant_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.train_X = x
        else:
            x = dm.test_X
            for index in self.constant_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.test_X = x
        return dm


class VarianceRemoverOperator(Operator):
    def __init__(self, params=5e-5):
        '''
        :param params: variance threshold
        '''
        super().__init__(DATA_PERPROCESSING, 'dp_variance_remover', params)
        self.low_variance_indices = []

    def operate(self, dm_list: typing.List, phase='train'):
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numericial_index = [i for i in range(len(feature_types))
                            if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        if phase == 'train':
            x = dm.train_X
            for index in numericial_index:
                feature = x[:, index]
                if feature.var() < self.params:  # Low-variance feature
                    self.low_variance_indices.append(index)
            self.low_variance_indices.reverse()
            for index in self.low_variance_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.train_X = x
        else:
            x = dm.test_X
            for index in self.low_variance_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.test_X = x
        return dm


class IdenticalRemoverOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_constant_remover', params)
        self.identical_indices = []

    def operate(self, dm_list: typing.List, phase='train'):
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numericial_index = [i for i in range(len(feature_types))
                            if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        if phase == 'train':
            x = dm.train_X
            for i, index in enumerate(numericial_index):
                feature = x[:, index]
                for ano_index in numericial_index[:i]:
                    ano_feature = x[:, ano_index]
                    if (feature == ano_feature).all():
                        self.identical_indices.append(i)
                        break
            self.identical_indices.reverse()
            for index in self.identical_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.train_X = x
        else:
            x = dm.test_X
            for index in self.identical_indices:
                del (dm.feature_types[index])
                x = np.delete(x, [index], axis=1)
            dm.test_X = x
        return dm
