import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from typing import Optional, List
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Selection of categorical and numerical features
numerical_columns = [key for key in data.keys() if data[key].dtype in ("int64", "float64")]
numerical_columns.remove('SalePrice')

# Filling missing values
def fillnan(data, col):
    for i in col:
        data[i].fillna(data[i].median(), inplace=True)
fillnan(data, numerical_columns)
fillnan(test, numerical_columns)
cotegorical_columns = [key for key in data.keys() if data[key].dtype == "object"]
nan_todrop = data.columns[data.isna().any()].tolist()
cotegorical_columns = [col for col in cotegorical_columns if col not in nan_todrop]

# Генерация новых признаков
def feat_gen(data):
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['SinceRemodel'] = data['YrSold'] - data['YearRemodAdd']
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
    data['TotalPorchArea'] = (data['WoodDeckSF'] + data['OpenPorchSF'] +
                            data['EnclosedPorch'] + data['3SsnPorch'] +
                            data['ScreenPorch'])
    data['GrLivAreaPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + 1)
    data['LotAreaPerFrontage'] = data['LotArea'] / (data['LotFrontage'] + 1e-10)
    data['OverallQual_GrLivArea'] = data['OverallQual'] * data['GrLivArea']
    data['GarageAge'] = data['YrSold'] - data['GarageYrBlt']
    data['BasementFinishRatio'] = (data['BsmtFinSF1'] + data['BsmtFinSF2']) / (data['TotalBsmtSF'] + 1e-10)

feat_gen(data)
feat_gen(test)
new_fetures = ['HouseAge', 'SinceRemodel', 'TotalSF', 'TotalPorchArea', 'GrLivAreaPerRoom',
               'LotAreaPerFrontage', 'OverallQual_GrLivArea', 'GarageAge', 'BasementFinishRatio']
numerical_columns.extend(new_fetures)

# Features requiring logarithm transformation
columns_to_log = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'LotAreaPerFrontage']
numerical_columns = [col for col in numerical_columns if col not in columns_to_log]

# Splitting the dataset
data_train, data_test, y_train, y_test = train_test_split(
    data[data.columns.drop('SalePrice')], data['SalePrice'], test_size=0.2, random_state=24
)

# Processing numerical features
class DataPreprocceser(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns:Optional[List[str]] = None,
                 columns_to_log:Optional[List[str]] = None):
        self.data = None
        self.scaler = StandardScaler()
        self.numerical_columns = numerical_columns
        self.columns_to_log = columns_to_log
    def fit(self, data):
        data = data[self.numerical_columns]
        self.scaler.fit(data)
        return self
    def transform(self, data):
        num_col = self.scaler.transform(data[self.numerical_columns])
        col_to_log = np.log1p(data[self.columns_to_log])
        return np.hstack([num_col, col_to_log])

# Processing categorical features
class OneHotPreprocessor(DataPreprocceser):
    def __init__(self, numerical_columns: Optional[List[str]] = None,
                 cotegorical_columns: Optional[List[str]] = None, **kwargs):
        super(OneHotPreprocessor, self).__init__(numerical_columns=numerical_columns, columns_to_log = columns_to_log)
        self.cotegorical_columns = cotegorical_columns
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, data, *args):
        super().fit(data)
        self.ohe.fit(data[self.cotegorical_columns])
        return self

    def transform(self, data):
        cont_transformed = super().transform(data)
        cat_transformed = self.ohe.transform(data[self.cotegorical_columns])
        return np.hstack([cont_transformed, cat_transformed])

# Training the model with L1 regularization and logarithm transformation of the target feature
class ExponentialLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self,*args, **kwargs):
        self.model  = linear_model.Lasso(**kwargs)

    def fit(self, X, Y):
        y_log = np.log(np.clip(Y, a_min =1e-10, a_max=None))
        self.model.fit(X, y_log)
        return self

    def predict(self, X):
        y_pred_log = self.model.predict(X)
        y_pred = np.exp(y_pred_log)
        return y_pred

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self
# Initializing the pipeline
def make_ultimate_pipeline():
    pipeline = Pipeline([
        ('preprocessor', OneHotPreprocessor(
            numerical_columns=numerical_columns,
            columns_to_log = columns_to_log,
            cotegorical_columns=cotegorical_columns
        )),
        ('regressor', ExponentialLinearRegression())
    ])
    return pipeline

pipeline = make_ultimate_pipeline()
pipeline.fit(data, data['SalePrice'])

# Searching for the best model using GridSearchCV
rmsle_scorer = make_scorer(mean_squared_error, greater_is_better=False)
kf = KFold(n_splits=5, shuffle=True, random_state=24)
grid = GridSearchCV(pipeline,
                    param_grid = {'regressor__alpha':np.logspace(-3, 3, num=7, base=10.)} ,
                    scoring = rmsle_scorer,
                    cv=kf
                    )
grid.fit(data, data['SalePrice'])
predictions = grid.predict(test)

#Output of results
test['SalePrice'] = predictions
submit = test[['Id', 'SalePrice']]
submit.to_csv('submit.csv', index=False)





