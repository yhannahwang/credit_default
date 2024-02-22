import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

# Assuming 'df' is your DataFrame with mixed data types (categorical and numerical)
# Preprocessing
# Convert categorical variables to numerical representations (e.g., one-hot encoding or label encoding)
# Normalize or standardize numerical variables if necessary
data_path = os.getcwd() + '/data/credit_card_default.csv'
df_credit = pd.read_csv(data_path,index_col='ID')
#X_train = df_credit.values

# Define a function to preprocess data (e.g., one-hot encoding for categorical variables)
def preprocess_data(preprocessed_df):
    # Your preprocessing code here
    return preprocessed_df

# Fit KNN model on preprocessed data
def fit_knn_model(df, n_neighbors=5):
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(df)
    return knn_model

# Generate synthetic data using KNN
def generate_synthetic_data(df, knn_model, num_samples=1000):
    synthetic_data = []
    for _, row in df.iterrows():
        # Find k-nearest neighbors
        distances, indices = knn_model.kneighbors([row])
        # Randomly sample from nearest neighbors
        synthetic_data.append(df.iloc[np.random.choice(indices[0], size=1)].values)
    return np.vstack(synthetic_data)

# Postprocessing
# Convert numerical representations back to categorical variables if needed
# Inverse normalize or standardize numerical variables if necessary

# Assuming 'df' is your original DataFrame
# Preprocess data
preprocessed_df = preprocess_data(df_credit)

# Fit KNN model
knn_model = fit_knn_model(preprocessed_df)

# Generate synthetic data
synthetic_data = generate_synthetic_data(preprocessed_df, knn_model, 2000)

# Convert synthetic data to DataFrame if needed
synthetic_df = pd.DataFrame(synthetic_data, columns=preprocessed_df.columns)

synthetic_df.to_csv(os.getcwd() +'/knn.csv', index=False)