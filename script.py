import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.impute import KNNImputer

# Load the data from a JSON file and save it as a CSV
data: pd.DataFrame = pd.read_json('data/final_dataset.json')
df: pd.DataFrame = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

# Drop unnecessary columns and rows with missing values
df.drop(['Url', 'Fireplace', 'Furnished', 'MonthlyCharges', 'Country', 'RoomCount', 'Locality', 'PropertyId', 'ConstructionYear'], axis=1, inplace=True)
df.dropna(subset=['Region', 'Province', 'District'], inplace=True)

# Function to calculate the boundaries for outliers
def remove_outliers(col: pd.Series) -> tuple[float, float]:
    Q1: float = col.quantile(0.15)
    Q3: float = col.quantile(0.85)
    IQR: float = Q3 - Q1
    lower_boundary: float = Q1 - 1.5 * IQR
    upper_boundary: float = Q3 + 1.5 * IQR
    return lower_boundary, upper_boundary

# Remove outliers
columns_with_outliers: list[str] = ['BathroomCount', 'BedroomCount', 'GardenArea', 'LivingArea', 'NumberOfFacades', 'Price', 'SurfaceOfPlot', 'ToiletCount']
for col in columns_with_outliers:
    lower_boundary, upper_boundary = remove_outliers(df[col])
    df = df[~((df[col] < lower_boundary) | (df[col] > upper_boundary))]

# Fill missing GardenArea values based on the difference between SurfaceOfPlot and LivingArea
for i, row in df.iterrows():
    if pd.isna(row['GardenArea']):
        if row['SurfaceOfPlot'] > row['LivingArea']:
            df.at[i, 'GardenArea'] = row['SurfaceOfPlot'] - row['LivingArea']

# Handle remaining missing values in GardenArea and add a Garden indicator column
for i, row in df.iterrows():
    if pd.isna(row['GardenArea']):
        df.at[i, 'Garden'] = 0
        df.at[i, 'GardenArea'] = 0
    else:
        df.at[i, 'Garden'] = 1

# Cap the NumberOfFacades at 4 and fill missing values with the mean
df['NumberOfFacades'] = df['NumberOfFacades'].apply(lambda x: 4 if x >= 4 else x)
df['NumberOfFacades'].fillna(round(df['NumberOfFacades'].mean(), 2), inplace=True)

# Fill missing values for SwimmingPool and Terrace
df['SwimmingPool'].fillna(0, inplace=True)
df['Terrace'].fillna(0, inplace=True)

# Encode categorical columns
ohe: OneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
ohetransform: pd.DataFrame = ohe.fit_transform(df[['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District']])
df = pd.concat([df, ohetransform], axis=1).drop(columns=['Kitchen', 'Province', 'TypeOfSale', 'FloodingZone', 'PEB', 'Region', 'SubtypeOfProperty', 'StateOfBuilding', 'District'])

# Impute missing values for selected numerical columns using KNN imputer
str_to_bit: list[str] = ['BathroomCount', 'LivingArea', 'NumberOfFacades', 'ToiletCount', 'SurfaceOfPlot', 'ShowerCount']
imputer: KNNImputer = KNNImputer(n_neighbors=5)
df[str_to_bit] = imputer.fit_transform(df[str_to_bit])

# Define CatBoost model parameters
params: dict[str, float] = {
    'learning_rate': 0.0921459262053085,
    'depth': 10,
    'subsample': 0.617993504291168,
    'colsample_bylevel': 0.8692370930849208,
    'min_data_in_leaf': 84,
}

# Split the target variable (Price) and features
y: np.ndarray = np.array(df['Price']).reshape(-1, 1)
X: np.ndarray = np.array(df.drop(['Price'], axis=1))

# Split the data into training and test sets
X_train: np.ndarray
X_test: np.ndarray
y_train: np.ndarray
y_test: np.ndarray
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

# Initialize and train the CatBoost model
model: CatBoostRegressor = CatBoostRegressor(**params, n_estimators=2662, loss_function='RMSE')
model.fit(X_train, y_train, verbose=100)

# Print the model score and mean absolute error on the test set
print(model.score(X_test, y_test))
y_pred: np.ndarray = model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
