import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from typing import Optional, List
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Selection of categorical and numerical features
numerical_columns = [key for key in data.keys() if data[key].dtype in ("int64", "float64")]
numerical_columns.remove('SalePrice')


# Filling missing values
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_with_zero = ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2',
                               'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',
                               'GarageArea', 'GarageCars', 'MasVnrArea']
        self.fill_with_na = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                             'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                             'GarageType', 'GarageQual', 'GarageCond', 'GarageFinish',
                             'PoolQC', 'Fence', 'MiscFeature']
        self.fill_with_specific = {
            'Electrical': 'SBrkr',  # Standard Circuit Breakers
            'MSZoning': 'RL',  # Residential Low Density
            'Utilities': 'AllPub',  # All public utilities
            'KitchenQual': 'TA',  # Typical/Average
            'Functional': 'Typ',  # Typical
            'MasVnrType': 'None',  # No masonry veneer
            'SaleType': 'WD'  # Warranty Deed - Conventional
        }

        # Will store computed values during fit - trailing underscore by convention
        self.mode_values_ = {}
        self.neighborhood_medians_ = None

    def fit(self, X, y=None):
        # Compute mode for mode columns
        for col in ['Exterior1st', 'Exterior2nd']:
            self.mode_values_[col] = X[col].mode()[0]

        # Compute neighborhood medians for LotFrontage
        self.neighborhood_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()

        return self

    def transform(self, X):
        X = X.copy()

        # Fill zeros
        X[self.fill_with_zero] = X[self.fill_with_zero].fillna(0)

        # Fill NA strings
        X[self.fill_with_na] = X[self.fill_with_na].fillna('NA')

        # Fill with specific values
        for col, value in self.fill_with_specific.items():
            X[col] = X[col].fillna(value)

        # Fill with computed modes
        for col in self.mode_values_:
            X[col] = X[col].fillna(self.mode_values_[col])

        # Fill LotFrontage using neighborhood medians
        for idx in X.index:
            if pd.isna(X.loc[idx, 'LotFrontage']):
                neighborhood = X.loc[idx, 'Neighborhood']
                X.loc[idx, 'LotFrontage'] = self.neighborhood_medians_[neighborhood]

        # Fill GarageYrBlt using YearBuilt (if available, otherwise 0)
        for idx in X.index:
            if pd.isna(X.loc[idx, 'GarageYrBlt']):
                if X.loc[idx, 'YearBuilt']:
                    year_built = X.loc[idx, 'YearBuilt']
                    X.loc[idx, 'GarageYrBlt'] = year_built
                else:
                    X.loc[idx, 'GarageYrBlt'] = 0

        return X


imputation_pipe = Pipeline([
    ('custom_imputer', CustomImputer())
])
data = imputation_pipe.fit_transform(data)
test = imputation_pipe.fit_transform(test)

# Selection of categorical and numerical features
numerical_columns = [key for key in data.keys() if data[key].dtype in ("int64", "float64")]
numerical_columns.remove('SalePrice')
cotegorical_columns = [key for key in data.keys() if data[key].dtype == "object"]

# Create new features
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

# Searching for the best model using GridSearchCV
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
kf = KFold(n_splits=5, shuffle=True, random_state=24)

param = {
    'regressor__iterations': [10000],
    'regressor__l2_leaf_reg': np.logspace(-3, 1, num=5, base=10.)
}
grid_search = GridSearchCV(
    pipeline,
    param_grid=param,
    scoring=mse_scorer,
    cv=kf
)
grid_search.fit(data, data['SalePrice'])
predictions = grid_search.predict(test)

#Output of results
test['SalePrice'] = predictions
submit = test[['Id', 'SalePrice']]
submit.to_csv('submit.csv', index=False)