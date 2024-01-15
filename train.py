import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from scipy import stats
import numpy as np
from sklearn.model_selection import GridSearchCV

def print_metrics(model, X_train, y_train, X_val, y_val):
    
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    r_squared_val = r2_score(np.expm1(y_val), np.expm1(y_pred_val))  
    mse_val = mean_squared_error(np.expm1(y_val), np.expm1(y_pred_val)) 
    mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_val))
    mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred_val))

    mae_train = mean_absolute_error(np.expm1(y_train), np.expm1(y_pred_train))
    mape_train = mean_absolute_percentage_error(np.expm1(y_train), np.expm1(y_pred_train))

    print(f"Mean Absolute Error (Train): {mae_train:.2f}")
    print(f"Mean Absolute Error Percentage (Train): {mape_train:.2f}")
    print("--------------------------------------")

    print(f"R-squared (Validation): {r_squared_val:.2f}")
    print(f"Mean Squared Error (Validation): {mse_val:.2f}")
    print(f"Mean Absolute Error (Validation): {mae:.2f}")
    print(f"Mean Absolute Error Percentage (Validation): {mape:.2f}")

print("1. Load dataset")
df = pd.read_csv('data/jobs_in_data.csv')

print("2. Filter duplicated rows")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("3. Drop outliers")
# Calculate Z-scores for 'salary_in_usd'
z_scores = stats.zscore(df['salary_in_usd'])
# Define a threshold (e.g., 3 standard deviations)
threshold = 3
# Find the indices of outliers
outlier_indices = np.where(np.abs(z_scores) > threshold)
# Filter the DataFrame to get rows with outliers
outliers = df.iloc[outlier_indices].sort_values(by='salary_in_usd', ascending=False)
df.drop(outliers.index, inplace=True)

print("4. Replace countries not in the top 10 with 'Rare'")
top_employee_10_countries = df['employee_residence'].value_counts().nlargest(10).index.tolist()
top_company_10_countries = df['company_location'].value_counts().nlargest(10).index.tolist()
df['employee_residence'] = df['employee_residence'].apply(lambda x: x if x in top_employee_10_countries else 'Rare')
df['company_location'] = df['company_location'].apply(lambda x: x if x in top_company_10_countries else 'Rare')
df

print("5. Features extraction")
# Splitting the dataset into 70% train, 15% validation, and 15% test
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

test_copy = test.copy()

# Label Encoder
label_encoder = LabelEncoder()
train['work_year'] = label_encoder.fit_transform(train['work_year'])
val['work_year'] = label_encoder.transform(val['work_year'])
test['work_year'] = label_encoder.transform(test['work_year'])

# One Hot Encoder
categorical_cols = ['job_category', 'employee_residence', 'experience_level', 
                     'work_setting', 'company_location', 'company_size']

one_hot_encoder = OneHotEncoder()
encoded_columns = one_hot_encoder.fit_transform(train[categorical_cols]).toarray()
encoded_col_names = one_hot_encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_columns, columns=encoded_col_names, index=train.index)
train = pd.concat([train, encoded_df], axis=1)

encoded_val_columns = one_hot_encoder.transform(val[categorical_cols]).toarray()
encoded_val_col_names = one_hot_encoder.get_feature_names_out(categorical_cols)
encoded_val = pd.DataFrame(encoded_val_columns, columns=encoded_val_col_names, index=val.index)
val = pd.concat([val, encoded_val], axis=1)

encoded_test_columns = one_hot_encoder.transform(test[categorical_cols]).toarray()
encoded_test_col_names = one_hot_encoder.get_feature_names_out(categorical_cols)
encoded_test = pd.DataFrame(encoded_test_columns, columns=encoded_test_col_names, index=test.index)
test = pd.concat([test, encoded_test], axis=1)

train.drop(columns=['job_title', 'salary_currency', 'salary','job_category', 'employee_residence', 'experience_level', 'employment_type',
                    "work_setting", 'company_location','salary', 'company_size'], axis=1, inplace=True )
val.drop(columns=['job_title', 'salary_currency','salary',  'job_category', 'employee_residence', 'experience_level', 'employment_type',
                    "work_setting", 'company_location', 'company_size'], axis=1, inplace=True )
test.drop(columns=['job_title', 'salary_currency','salary',  'job_category', 'employee_residence', 'experience_level', 'employment_type',
                    "work_setting", 'company_location', 'company_size'], axis=1, inplace=True )

print("Final shape:")
print("train", train.shape)
print("val", val.shape)
print("test", test.shape)

X_train = train.drop(['salary_in_usd'], axis=1) 
y_train = np.log1p(train['salary_in_usd'])
X_val = val.drop(['salary_in_usd'], axis=1)
y_val = np.log1p(val['salary_in_usd'])
X_test = test.drop(['salary_in_usd'], axis=1)
y_test = np.log1p(test['salary_in_usd'])

print("Tuning CatBoostRegressor.......")
catboost_model = CatBoostRegressor(verbose=False, random_state=42)

# Define the hyperparameters to search
param_grid = {
    'iterations': [100, 200, 300],  # Number of boosting iterations
    'depth': [6, 8, 10],            # Depth of the trees
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
}

grid_search_catboost = GridSearchCV(catboost_model, param_grid, cv=5, scoring='neg_mean_absolute_error')

grid_search_catboost.fit(X_train, y_train)

best_params_catboost = grid_search_catboost.best_params_
best_catboost_model = grid_search_catboost.best_estimator_

print("Best hyperparameters for CatBoostRegressor:")
print(best_params_catboost)

print("Train CatBoostRegressor with best hyperparameters:")
# Initialize the CatBoostRegressor model with the best hyperparameters
catboost_model = CatBoostRegressor(iterations=best_params_catboost['iterations'],
                                   depth=best_params_catboost['depth'],
                                   learning_rate=best_params_catboost['learning_rate'],
                                   verbose=False, random_state=42)

catboost_model.fit(X_train, y_train)

print_metrics(catboost_model, X_train, y_train, X_test, y_test)

print("Save the model:")
with open('models_binary/catboost_model.pkl', 'wb') as f_model:
    pickle.dump(catboost_model, f_model)

# Save the LabelEncoder
with open('models_binary/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save the test part of the original dataset
with open('models_binary/one_hot_encoder.pkl', 'wb') as f:
    pickle.dump(one_hot_encoder, f)

test_copy.to_csv('models_binary/test_copy.csv')