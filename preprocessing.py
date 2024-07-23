import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def json_to_csv():
    data = pd.read_json('final_dataset.json') 
    df = pd.DataFrame(data)
    df.to_csv('dataset.csv', index=False)
    return df

def drop_col(df):
    df.drop(['Url', 'Fireplace', 'Furnished', 'MonthlyCharges', 'Country', 'RoomCount', 'Locality', 'PropertyId', 'ConstructionYear'], axis=1, inplace=True)
    df.dropna(subset=['Region','Province', 'District'], inplace=True)

def remove_outliers(col, df):
    Q1 = col.quantile(0.01)
    Q3 = col.quantile(0.99)
    IQR = Q3 - Q1
    lower_boundary = Q1 - 1.5 * IQR
    upper_boundary = Q3 + 1.5 * IQR

    columns_with_outliers = df[['BathroomCount','BedroomCount','GardenArea','LivingArea','NumberOfFacades','Price','SurfaceOfPlot','ToiletCount',]]
    for col in columns_with_outliers:
        lower_boundary, upper_boundary = remove_outliers(df[col])
        df = df[~((df[col] < (lower_boundary)) |(df[col] > (upper_boundary)))]
    return lower_boundary, upper_boundary

def replace_nan(df):
    df[['BathroomCount', 'ShowerCount']] = df[['BathroomCount', 'ShowerCount']].fillna(0)
    df['BathroomCount'] = df['BathroomCount'] + df['ShowerCount']


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


    df.loc[df['BathroomCount'] < 1, 'BathroomCount'] = round(df['BathroomCount'].mean(), 2)

    df['LivingArea'].fillna(round(df['LivingArea'].mean(), 2), inplace=True)

    df['NumberOfFacades'] = df['NumberOfFacades'].apply(lambda x: 4 if x >= 4 else x)

    df['NumberOfFacades'].fillna(round(df['NumberOfFacades'].mean(), 2), inplace=True)

    df['ToiletCount'].fillna(round(df['ToiletCount'].mean(), 2), inplace=True)

    df['SurfaceOfPlot'].fillna(round(df['SurfaceOfPlot'].mean(), 2), inplace=True)

    df['SwimmingPool'].fillna(0, inplace=True)

    df['Terrace'].fillna(0, inplace=True)
    return df

def str_to_int():
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    ohetransform = ohe.fit_transform(df[['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District']])
    df = pd.concat([df,ohetransform],axis=1).drop(columns=['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District'])
    return df

# df.to_csv('datasett.csv', index=False)

