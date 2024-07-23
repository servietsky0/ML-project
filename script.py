import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.impute import KNNImputer 


data = pd.read_json('data/final_dataset.json') 
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

df.drop(['Url', 'Fireplace', 'Furnished', 'MonthlyCharges', 'Country', 'RoomCount', 'Locality', 'PropertyId', 'ConstructionYear'], axis=1, inplace=True)
df.dropna(subset=['Region','Province', 'District'], inplace=True)


def remove_outliers(col):
    Q1 = col.quantile(0.15)
    Q3 = col.quantile(0.85)
    IQR = Q3 - Q1
    lower_boundary = Q1 - 1.5 * IQR
    upper_boundary = Q3 + 1.5 * IQR
    return lower_boundary, upper_boundary


columns_with_outliers = df[['BathroomCount','BedroomCount','GardenArea','LivingArea','NumberOfFacades','Price','SurfaceOfPlot','ToiletCount',]]
for col in columns_with_outliers:
    lower_boundary, upper_boundary = remove_outliers(df[col])
    df = df[~((df[col] < (lower_boundary)) |(df[col] > (upper_boundary)))]


for i, row in df.iterrows():
    if pd.isna(row['GardenArea']):
        if row['SurfaceOfPlot'] > row['LivingArea']:
            df.at[i,'GardenArea'] = row['SurfaceOfPlot'] - row['LivingArea']

for i, row in df.iterrows():
    if pd.isna(row['GardenArea']):
        df.at[i,'Garden'] = 0
        df.at[i,'GardenArea'] = 0
    else:
        df.at[i,'Garden'] = 1



df['NumberOfFacades'] = df['NumberOfFacades'].apply(lambda x: 4 if x >= 4 else x)

df['NumberOfFacades'].fillna(round(df['NumberOfFacades'].mean(), 2), inplace=True)

df['SwimmingPool'].fillna(0, inplace=True)

df['Terrace'].fillna(0, inplace=True)


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
ohetransform = ohe.fit_transform(df[['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District']])
df = pd.concat([df,ohetransform],axis=1).drop(columns=['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District'])


str_to_bit = ['BathroomCount', 'LivingArea', 'NumberOfFacades', 'ToiletCount', 'SurfaceOfPlot', 'ShowerCount']
imputer = KNNImputer(n_neighbors=5)
df[str_to_bit] = imputer.fit_transform(df[str_to_bit])


df.to_csv('datasett.csv', index=False)
df = pd.read_csv('datasett.csv')


params = {
        'learning_rate': 0.0921459262053085,
        'depth': 10,
        'subsample': 0.617993504291168,
        'colsample_bylevel': 0.8692370930849208,
        'min_data_in_leaf': 84,
    }


y = np.array(df['Price']).reshape(-1, 1)
X = np.array(df.drop(['Price'], axis=1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

model = CatBoostRegressor(**params, loss_function='RMSE')
model.fit(X_train, y_train, verbose=100)


print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


df.to_csv('datasett.csv', index=False)

