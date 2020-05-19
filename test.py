from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from math import sqrt

repo = ('C:/Users/Ben Noyes/Desktop/Atom/Atom/Housing Prices/Datasets/housing_prices_trainfile_inc_onehot.csv')
df_train = pd.read_csv(repo, skipinitialspace=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# print(df_train.isnull().sum())


repo = ('C:/Users/Ben Noyes/Desktop/Atom/Atom/Housing Prices/Datasets/test.csv')
df_test = pd.read_csv(repo, skipinitialspace=False)

df_test = df_test.drop(['Street', 'Alley', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Heating', 'GarageYrBlt',
                        'HeatingQC', 'FireplaceQu', 'CentralAir', 'Electrical', 'LowQualFinSF', 'BsmtHalfBath', 'Functional', 'PavedDrive', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold'], 1)

df_test.isnull().sum()
# investigate nulls

df = df_test

# 'MSZoning', MCAR
df[df.MSZoning.isnull() == True]
df[455:460]

# 'LotFrontage', MAR

# 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
# some MNAR, some MAR

df[df.BsmtCond.isnull() == True][['BsmtQual', 'BsmtCond', 'BsmtExposure',
                                  'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]

df.BsmtFinSF2 = df.BsmtFinSF2.fillna(0)
df.BsmtUnfSF = df.BsmtUnfSF.fillna(0)
df.TotalBsmtSF = df.TotalBsmtSF.fillna(0)
df.BsmtFinType2 = df.BsmtFinType2.fillna('None')
df.BsmtFinSF1 = df.BsmtFinSF1.fillna(0)

df.at[580, 'BsmtCond'] = 'Gd'
df.at[725, 'BsmtCond'] = 'TA'
df.at[1064, 'BsmtCond'] = 'TA'

df.BsmtQual = df.BsmtQual.fillna('None')
df.BsmtCond = df.BsmtCond.fillna('None')
df.BsmtExposure = df.BsmtExposure.fillna('None')
df.BsmtFinType1 = df.BsmtFinType1.fillna('None')


# 'BsmtFullBath', MAR
df[df.BsmtFullBath.isnull() == True]
df.BsmtFullBath = df.BsmtFullBath.fillna(0)


# 'KitchenQual', MCAR
df[df.KitchenQual.isnull() == True]
df.KitchenQual = df.KitchenQual.fillna('TA')

df.isnull().sum()

# 'Garage', MAR and MNAR

df[(df.GarageFinish.isnull() == True) & (df.GarageType.isnull() == False)][['GarageType',
                                                                            'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]

df[(df.GarageFinish.isnull() == True) & (df.GarageType.isnull() == False)].GarageFinish.fillna('TA')
df[(df.GarageFinish.isnull() == True) & (df.GarageType.isnull() == False)].GarageQual.fillna('TA')
df[(df.GarageFinish.isnull() == True) & (df.GarageType.isnull() == False)].GarageCond.fillna('TA')

df[1110:1120][['GarageType',
               'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
df[(df.GarageFinish.isnull() == True) & (df.GarageType.isnull() == False)][['GarageType',
                                                                            'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
df.at[1116, 'GarageFinish'] = 'Unf'
df.at[1116, 'GarageCars'] = 1
df.at[1116, 'GarageArea'] = 100
df.at[1116, 'GarageQual'] = 'TA'
df.at[1116, 'GarageCond'] = 'TA'

df[(df.GarageFinish.isnull() == True) & (df.GarageArea > 0)]

df.at[666, 'GarageFinish'] = 'Unf'
df.at[666, 'GarageQual'] = 'TA'
df.at[666, 'GarageCond'] = 'TA'

df[df.GarageFinish.isnull() == True]

df.GarageType = df.GarageType.fillna('None')
df.GarageFinish = df.GarageFinish.fillna('None')
df.GarageQual = df.GarageQual.fillna('None')
df.GarageCond = df.GarageCond.fillna('None')

df.isnull().sum()

# add conditional variables

df['FancyCulDSac'] = ((df.LotConfig == 'CulDSac') & (df.LotShape == 'IR2'))

df['PoorExtCond'] = df.ExterCond == 'Fa'

df['PoorBsmtCond'] = df.BsmtCond == 'Fa'

df['SlabFoundation'] = df.Foundation == 'Slab'
df['CBlockFoundation'] = (df.Foundation == 'CBlock') | (df.Foundation == 'BrkTil')
df['StrongFoundation'] = (df.Foundation == 'Stone') | (df.Foundation == 'Wood')

df['ExKitchen'] = df.KitchenQual == 'Ex'
df['GdKitchen'] = df.KitchenQual == 'Gd'
df['TAKitchen'] = df.KitchenQual == 'TA'

df['OffSaleType'] = df.SaleType.isin(['COD', 'ConLD', 'Oth'])

df['BadGarage'] = (df.GarageCond == 'None') | (df.GarageCond == 'Fa') | (
    df.GarageQual == 'None') | (df.GarageQual == 'Fa') | (df.GarageQual == 'Po')
df['UnfGarage'] = df.GarageFinish.isin(['Unf'])
df['FinGarage'] = df.GarageFinish.isin(['Fin', 'RFn'])

df['New2Story'] = df.MSSubClass.isin([60])
df['Old1Story'] = df.MSSubClass.isin([30])

df['FVResidential'] = df.MSZoning.isin(['FV'])

# sum column dtypes by dataframe

df_categoricals = df[['LotConfig', 'LotShape',
                      'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual']]

df_numerical = df[['Id', 'BedroomAbvGr', 'OverallQual', 'LotFrontage', 'LotArea', 'WoodDeckSF', 'OpenPorchSF', 'GarageArea',
                   'GarageCars', '1stFlrSF', 'Fireplaces', 'TotRmsAbvGrd', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', ]]

df_bool = df[['Id', 'New2Story', 'Old1Story', 'FVResidential', 'FancyCulDSac', 'PoorExtCond', 'SlabFoundation', 'CBlockFoundation',
              'StrongFoundation', 'PoorBsmtCond', 'ExKitchen', 'GdKitchen', 'TAKitchen', 'BadGarage', 'UnfGarage', 'FinGarage', 'OffSaleType']]

df_categoricals.KitchenQual.value_counts()

len(df_categoricals)

# Encode Categoricals with HotEncoder
onehot = OneHotEncoder(sparse=True)
df_categoricals_encoded = pd.DataFrame(onehot.fit_transform(df_categoricals).toarray(), columns=[
    'Corner_LotConf', 'CulDSac_LotConf', 'FR2_LotConf', 'FR3_LotConf', 'Inside_LotConf',
    'IR1_LotSh', 'IR2_LotSh', 'IR3_LotSh', 'Reg_LotSh',
    'Ex_ExtQual', 'Fa_ExtQual', 'Gd_ExtQual', 'TA_ExtQual',
    'Ex_BsmtQual', 'Fa_BsmtQual', 'Gd_BsmtQual', 'None_BsmtQual', 'TA_BsmtQual',
    'Av_BsmtExp', 'Gd_BsmtExp', 'Mn_BsmtExp', 'No_BsmtExp', 'None_BsmtExp',
    'Ex_Kitch', 'Fa_Kitch', 'Gd_Kitch', 'TA_Kitch'
])

df_categoricals_encoded['Id'] = df['Id']

df_all_test = pd.concat([df_categoricals_encoded, df_bool, df_numerical], 1)

df_all_test = df_all_test.loc[:, df_all_test.columns.duplicated() == False]

df_all_test.isnull().sum()

# merge with train, check shapes

df_train.isnull().sum()

df_train_no_sales = df_train.drop(['SalePrice', 'SaleBins'], 1)
df_train_no_sales.columns
df_all_test.columns

df_train_sales = df_train[['Id', 'SalePrice']]

df_lotfrontage = df_train_no_sales.append(df_all_test)

pd.set_option('display.max_columns', None)

df_lotfrontage.LotFrontage = df_lotfrontage.LotFrontage.fillna(
    df_lotfrontage.LotFrontage.interpolate(method='linear'))

print(df_lotfrontage.isnull().any().sum())

print(len(df_lotfrontage))

# add prediction columns

df_predict = pd.merge(df_lotfrontage, df_train_sales, how='outer', on='Id')

print(df_predict.shape)

iterative = IterativeImputer()
prediction = pd.DataFrame(iterative.fit_transform(df_predict), columns=df_predict.columns)

# formatting

final_prediction = prediction[['Id', 'SalePrice']][-1459:]

print(final_prediction.shape)
print(final_prediction[:5])
print(final_prediction[-5:])
final_prediction['Id'] = final_prediction['Id'].astype(np.int64)

file = 'housing_prices_onehot.csv'

final_prediction.to_csv(file, index=False)
