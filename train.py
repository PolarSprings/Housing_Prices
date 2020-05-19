from sklearn.preprocessing import OneHotEncoder
import sys
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot
import plotly.io as pio
import numpy as np
import pandas as pd
from math import sqrt

repo = ('C:/Users/Ben Noyes/Desktop/Atom/Atom/Housing Prices/Datasets/train.csv')
df = pd.read_csv(repo, skipinitialspace=False)

pd.set_option('display.max_columns', None)

# check for nulls
df.isnull().sum()

# set threshold for rows with an egregious amount of missing data
# tighten threshold

len(df.index)
df.columns
# df = df.dropna(thresh=71)
# len(df.index)

# check if any categores are erroneous

df.loc[:, df.isnull().any()]
df.Alley[df.Alley.isnull() == False][:5]
df.PoolQC[df.PoolQC.isnull() == False][:5]
df.MiscFeature[df.MiscFeature.isnull() == False][:5]

# investigate which variables are Missing at Random (MAR)

# variable 1, 'LotFrontage', is MAR

df[df.LotFrontage.isnull() == True][:10]
df.loc[::7]

# varaible 2, 'Alley', is MNAR

df[df.Alley.isnull() == False][:5]
df.Alley = df.Alley.fillna('None')
df.loc[20:35]

# variables 3 and 4, 'MasVnrType', are MAR

df[df.MasVnrType.isnull() == True][:5]
df[df.MasVnrArea.isnull() == True][:5]

# variables 5 through 9, 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', is MNAR

df[df.BsmtQual.isnull() == True][:5]
df.loc[90:105]

len(df[df.BsmtQual.isnull() == True])
len(df[df.TotalBsmtSF == 0])

df.BsmtQual = df.BsmtQual.fillna('None')
df.BsmtCond = df.BsmtCond.fillna('None')
df.BsmtExposure = df.BsmtExposure.fillna('None')
df.BsmtFinType1 = df.BsmtFinType1.fillna('None')
df.BsmtFinType2 = df.BsmtFinType2.fillna('None')

df.loc[90:105]

# variable 10, 'Electrical' (MCAR)

df[df.Electrical.isnull() == True]
# Electrical_mode = df.Electrical.mode()[0]
# df['Electrical'] = df.Electrical.fillna(Electrical_mode)

# variable 11, 'FireplaceQu' (MNAR)

df[df.FireplaceQu.isnull() == True]
df[::5]

df.FireplaceQu = df.FireplaceQu.fillna('None')

# variables 12-16, 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond' (MNAR)

len(df[df.GarageType.isnull() == True])
df.loc[48:53]

df.GarageType = df.GarageType.fillna('None')
df.GarageType = df.GarageType.fillna('None')
df.GarageFinish = df.GarageFinish.fillna('None')
df.GarageQual = df.GarageQual.fillna('None')
df.GarageCond = df.GarageCond.fillna('None')

# df.GarageYrBlt.fillna(df.GarageYrBuit.interpolate(method='linear'))

# variable 17, 'PoolQC' (MNAR)

df[df.PoolQC.isnull() == False][:5]

df.loc[810:812]
df.PoolQC = df.PoolQC.fillna('None')

# variable 18, 'Fence', (MNAR)

df[df.Fence.isnull() == True]
df.loc[1455:1458]

df.Fence = df.Fence.fillna('None')

# variable 19, 'MiscFeature', (MNAR)

df[df.MiscFeature.isnull() == False]
df.loc[1230:1300]
df.MiscFeature = df.MiscFeature.fillna('None')

# sum the nulls, confirm data should remain as nulls
pd.set_option('display.max_rows', None)

np.mean(df.GarageYrBlt)
np.var(df.GarageYrBlt)
df.GarageYrBlt.size

df.Electrical.mode()
df.Electrical[::100]
df.Electrical = df.Electrical.fillna('SBrkr')

np.mean(df.LotFrontage)
np.var(df.LotFrontage)
df.LotFrontage.count()
df.LotFrontage.nunique()
df.LotFrontage.size

df.isnull().sum()

# Wait
# bucket sale price

SaleBins = pd.qcut(df.SalePrice, 20, labels=False)
SaleBins.value_counts()
df['SaleBins'] = SaleBins

df.SaleBins.value_counts()
# df.SalePrice[df.SaleBins == 1][:5]
# df.SalePrice[df.SaleBins == 10][:5]


# stats
df.info()

pd.set_option('display.max_rows', 10)

x = np.array(df.BedroomAbvGr)
y = np.array(df.SaleBins)


def sum_of_squares_residuals(x):
    SSx = np.sum(x**2) - (1/len(x))*((np.sum(x))**2)
    return SSx


def correlation_coefficient(x, y):
    r = np.sum((x - np.mean(x)) * (y - np.mean(y))) / \
        (sqrt((np.sum((x - np.mean(x))**2))) * sqrt((np.sum((y - np.mean(y))**2))))
    return r


r01 = correlation_coefficient(x, y)


def coefficient_of_determination(r):
    r2 = r ** 2
    return r2


r2_01 = coefficient_of_determination(r01)


x = np.array(df.OverallQual)
y = np.array(df.SaleBins)

r02 = correlation_coefficient(x, y)
r2_02 = coefficient_of_determination(r02)


# inclusion as necessary
x = np.array(df.LotFrontage[df.LotFrontage.isnull() == False])
y = np.array(df.SaleBins[df.LotFrontage.isnull() == False])

r03 = correlation_coefficient(x, y)

r2_03 = coefficient_of_determination(r03)


x = np.array(df.LotArea)
y = np.array(df.SaleBins)

r04 = correlation_coefficient(x, y)
r2_04 = coefficient_of_determination(r04)


x = np.array(df.WoodDeckSF)
y = np.array(df.SaleBins)

r05 = correlation_coefficient(x, y)
r2_05 = coefficient_of_determination(r05)


x = np.array(df.OpenPorchSF)
y = np.array(df.SaleBins)

r06 = correlation_coefficient(x, y)
r2_06 = coefficient_of_determination(r06)


x = np.array(df.GarageArea)
y = np.array(df.SaleBins)

r07 = correlation_coefficient(x, y)
r2_07 = coefficient_of_determination(r07)

x = np.array(df.GarageCars)
y = np.array(df.SaleBins)

r08 = correlation_coefficient(x, y)
r2_08 = coefficient_of_determination(r08)


df['HasGarage'] = df.GarageArea > 0

x = np.array(df.HasGarage)
y = np.array(df.SaleBins)

r09 = correlation_coefficient(x, y)
r2_09 = coefficient_of_determination(r09)


x = np.array(df['1stFlrSF'])
y = np.array(df.SaleBins)

r10 = correlation_coefficient(x, y)
r2_10 = coefficient_of_determination(r10)


x = np.array(df.Fireplaces)
y = np.array(df.SaleBins)

r11 = correlation_coefficient(x, y)
r2_11 = coefficient_of_determination(r11)


x = np.array(df.TotRmsAbvGrd)
y = np.array(df.SaleBins)

r12 = correlation_coefficient(x, y)
r2_12 = coefficient_of_determination(r12)

x = np.array(df.GrLivArea)
y = np.array(df.SaleBins)

r13 = correlation_coefficient(x, y)
r2_13 = coefficient_of_determination(r13)


df[['GrLivArea', 'TotRmsAbvGrd']][:5]

x = np.array(df.YearBuilt)
y = np.array(df.SaleBins)

r14 = correlation_coefficient(x, y)
r2_14 = coefficient_of_determination(r14)


x = np.array(df.YearRemodAdd)
y = np.array(df.SaleBins)

r15 = correlation_coefficient(x, y)
r2_15 = coefficient_of_determination(r15)


# charts

fig01 = make_subplots(rows=1, cols=2, specs=[
                      [{'type': 'bar'}, {'type': 'scatter'}]], subplot_titles=['Price by Quality', 'Price by Age'], shared_yaxes=True)

fig01.add_bar(x=np.arange(0, 10),
              y=[np.mean(df.SalePrice[df.OverallQual == 1]), np.mean(df.SalePrice[df.OverallQual == 2]), np.mean(df.SalePrice[df.OverallQual == 3]), np.mean(df.SalePrice[df.OverallQual == 4]), np.mean(df.SalePrice[df.OverallQual == 5]), np.mean(
                  df.SalePrice[df.OverallQual == 6]), np.mean(df.SalePrice[df.OverallQual == 7]), np.mean(df.SalePrice[df.OverallQual == 8]), np.mean(df.SalePrice[df.OverallQual == 9]), np.mean(df.SalePrice[df.OverallQual == 10])],
              name='Quality',
              text='Overall Quality of Listing',
              row=1, col=1)

fig01.add_scatter(x=np.array(df.YearBuilt[df.YearBuilt > 0]),
                  y=np.array(df.SalePrice[df.YearBuilt > 0]),
                  mode='markers',
                  name='Age',
                  text='Year Built',
                  row=1, col=2)

fig01.update_layout({'title': {'text': 'What Impacts the Cost of a Home?'}})

fig01.update_xaxes({'title': {'text': 'Overall Quality'}}, row=1, col=1)
fig01.update_yaxes({'title': {'text': 'Sale Price'}}, row=1, col=1)
fig01.update_xaxes({'title': {'text': 'Year Built'}}, row=1, col=2)

plot(fig01)

# without overfitting confirm the variables to include in predict model
print('r2_01 is: ', r2_01)
print('r2_02 is: ', r2_02)
print('r2_03 is: ', r2_03)
print('r2_04 is: ', r2_04)
print('r2_05 is: ', r2_05)
print('r2_06 is: ', r2_06)
print('r2_07 is: ', r2_07)
print('r2_08 is: ', r2_08)
print('r2_09 is: ', r2_09)
print('r2_10 is: ', r2_10)
print('r2_11 is: ', r2_11)
print('r2_12 is: ', r2_12)
print('r2_13 is: ', r2_13)
print('r2_14 is: ', r2_14)
print('r2_15 is: ', r2_15)


# add conditional variables

pd.set_option('display.max_rows', None)

x = np.array(df.BsmtUnfSF)
y = np.array(df.SaleBins)

r16 = correlation_coefficient(x, y)
r2_16 = coefficient_of_determination(r16)

df[['BsmtQual', 'BsmtExposure', 'SalePrice']][:75]

fig02 = make_subplots(rows=1, cols=1, specs=[[{'type': 'Mesh3d'}]])

fig02.add_mesh3d(x=df.BsmtQual,
                 y=df.BsmtExposure,
                 z=np.array(df.SalePrice),
                 row=1, col=1)

plot(fig02)

# keep as unique variables

# lot variables

df[['LotConfig', 'LotShape', 'LandContour', 'SalePrice']][::20]


fig03 = make_subplots(rows=1, cols=1, specs=[[{'type': 'Mesh3d'}]])

fig03.add_mesh3d(x=df.LotConfig,
                 y=df.LotShape,
                 z=np.array(df.SaleBins),
                 row=1, col=1)

plot(fig03)

df['FancyCulDSac'] = ((df.LotConfig == 'CulDSac') & (df.LotShape == 'IR2'))

GroupB_LCLS = df.groupby(['FancyCulDSac']).agg({'SalePrice': np.mean})

# Exter, foundation

df[['ExterQual', 'ExterCond', 'Foundation', 'SalePrice']][:15]

df['PoorExtCond'] = df.ExterCond == 'Fa'

df.SalePrice[df.BsmtCond == 'Fa'].mean()
df.SalePrice[df.BsmtCond != 'Fa'].mean()

df['PoorBsmtCond'] = df.BsmtCond == 'Fa'

GroupB_F = df.groupby('Foundation').agg({'SalePrice': [np.mean, np.max, np.min, np.std]})

df['SlabFoundation'] = df.Foundation == 'Slab'

df['CBlockFoundation'] = (df.Foundation == 'CBlock') | (df.Foundation == 'BrkTil')

df['StrongFoundation'] = (df.Foundation == 'Stone') | (df.Foundation == 'Wood')

GroupB_F

# kitchen

df[['CentralAir', 'KitchenQual', 'SaleType', 'SaleCondition', 'SalePrice']][:15]
df.KitchenQual.value_counts()

GroupB_K = df.groupby('KitchenQual').agg(
    {'SalePrice': [np.mean, np.max, np.min, np.std]})

df['ExKitchen'] = df.KitchenQual == 'Ex'
df['GdKitchen'] = df.KitchenQual == 'Gd'
df['TAKitchen'] = df.KitchenQual == 'TA'

# Sale condition, sale Sale
df[['SaleType', 'SaleCondition', 'SalePrice']][:5]

GroupB_S = df.groupby(['SaleType']).agg(
    {'SalePrice': [np.mean, np.max, np.min, np.std, 'count']})

df['OffSaleType'] = df.SaleType.isin(['COD', 'ConLD', 'Oth'])

# Garage type
df[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'SalePrice']][:15]

GroupB_G = df.groupby(['GarageFinish']).agg(
    {'SalePrice': [np.mean, np.max, np.min, np.std, 'count']})

df['BadGarage'] = (df.GarageCond == 'None') | (df.GarageCond == 'Fa') | (
    df.GarageQual == 'None') | (df.GarageQual == 'Fa') | (df.GarageQual == 'Po')
df['UnfGarage'] = df.GarageFinish.isin(['Unf'])
df['FinGarage'] = df.GarageFinish.isin(['Fin', 'RFn'])

# confirm pre-categorized labels for housing types
df[['MSSubClass', 'MSZoning', 'SalePrice']][:15]

GroupB_MS = df.groupby(['MSZoning']).agg(
    {'SalePrice': [np.mean, np.max, np.min, np.std, 'count']})

df['New2Story'] = df.MSSubClass.isin([60])
df['Old1Story'] = df.MSSubClass.isin([30])

df['FVResidential'] = df.MSZoning.isin(['FV'])

# corr check for remaining numericals
x = np.array(df['2ndFlrSF'])
y = np.array(df.SalePrice)

r17 = correlation_coefficient(x, y)

x = np.array(df['BsmtFullBath'])

r18 = correlation_coefficient(x, y)

x = np.array(df['BsmtHalfBath'])

r19 = correlation_coefficient(x, y)

x = np.array(df['FullBath'])

r20 = correlation_coefficient(x, y)

x = np.array(df['HalfBath'])

r21 = correlation_coefficient(x, y)

x = np.array(df['LowQualFinSF'])

r22 = correlation_coefficient(x, y)

x = np.array(df['KitchenAbvGr'])

r23 = correlation_coefficient(x, y)

# sum column dtypes by dataframe
df['Id'] = df.index+1

df_categoricals = df[['LotConfig', 'LotShape',
                      'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual']]

df_numerical = df[['Id', 'BedroomAbvGr', 'OverallQual', 'LotFrontage', 'LotArea', 'WoodDeckSF', 'OpenPorchSF', 'GarageArea',
                   'GarageCars', '1stFlrSF', 'Fireplaces', 'TotRmsAbvGrd', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'SalePrice', 'SaleBins']]

df_bool = df[['Id', 'New2Story', 'Old1Story', 'FVResidential', 'FancyCulDSac', 'PoorExtCond', 'SlabFoundation', 'CBlockFoundation',
              'StrongFoundation', 'PoorBsmtCond', 'ExKitchen', 'GdKitchen', 'TAKitchen', 'BadGarage', 'UnfGarage', 'FinGarage', 'OffSaleType']]


print(df_categoricals.LotShape.value_counts())

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

df_categoricals_encoded[: 5]

# merge categoricals, bools, and numerical

df_all_train = pd.concat([df_categoricals_encoded, df_bool, df_numerical], 1)

df_all_train = df_all_train.loc[:, df_all_train.columns.duplicated() == False]

print(df_all_train[:5])

# print section
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

file = 'housing_prices_trainfile_inc_onehot.csv'

df_all_train.to_csv(file, index=False)
