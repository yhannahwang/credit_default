import os
import sys
import pandas as pd
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_helper import model_prep_pipleline,grid_search_param

data_path = os.getcwd() + '/data/credit_card_default.csv'
df_credit = pd.read_csv(data_path,index_col='ID')
# data type
df_credit.dtypes
df_credit.shape
# all int or float
df_credit.describe()
df_credit['default.payment.next.month'].value_counts()

# Example: Histogram of a numerical variable
num_i = 6
plt.figure(figsize=(8, 6))
#plt.hist(df_credit['PAY_AMT6'], bins=20, color='skyblue', edgecolor='black')
plt.scatter(df_credit['BILL_AMT'+str(num_i)], df_credit['PAY_AMT'+str(num_i)], color='orange', alpha=0.5)
plt.title('Histogram of BILL_AMT'+str(num_i)+'& PAY_AMT'+str(num_i))
plt.xlabel('BILL_AMT'+str(num_i))
plt.ylabel('PAY_AMT'+str(num_i))
plt.grid(True)
plt.show()

(1e2 * df_credit.isnull().sum()/len(df_credit)).plot(kind='barh')
plt.xlim(0, 10**2)
plt.grid();
# no missing value

# build model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

num_feat = df_credit.drop('default.payment.next.month', axis=1).select_dtypes(include=np.number).columns
X = df_credit.drop('default.payment.next.month', axis=1)
y = df_credit['default.payment.next.month']
# split train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# lr, rt, rf
classifier_lst = [LogisticRegression,DecisionTreeClassifier,RandomForestClassifier]
df_score = []
for m in classifier_lst:
    model_pre = model_prep_pipleline(X_train, y_train, m())
    y_pred_m = model_pre.predict_proba(X_test)[:, 1]
    print(roc_auc_score(y_test, y_pred_m))
    #lr: 0.7220387091719236
    #dt: 0.6199394567830337
    #rf: 0.7605566481372852


# xgb
# Define the grid of parameters to search
#param_grid = {
#    'max_depth': [ 4, 5],
#    'learning_rate': [0.1, 0.2,0.01],
#    'n_estimators': [50, 70,100],
#    'min_child_weight': [2, 3],
#    'subsample':[0.8,1]
#}
#best_model = grid_search_param(param_grid,model,X_train,y_train)

param_best = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 70,
    'min_child_weight': 3,
    'subsample':1
}
model_best = XGBClassifier(**param_best)
model_best.fit(X_train, y_train)
y_pred = model_best.predict_proba(X_test)[:,1]

# Evaluate the best model
auc = roc_auc_score(y_test, y_pred)
print("auc:", auc)
# 0.779

# precision & recall in different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Initialize an empty list to store DataFrames
dfs = []

# Calculate precision for each threshold
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the threshold
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)
    # Calculate precision
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    # Create a DataFrame for the current threshold and precision
    df = pd.DataFrame({'Threshold': [threshold], 'Precision': [precision],'Recall':[recall]})
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate the list of DataFrames into a single DataFrame
precision_recall_df = pd.concat(dfs, ignore_index=True)
# write the DataFrame
precision_recall_df.to_csv(os.getcwd() +'/precision_recall.csv', index=False)


# Get feature importance scores dictionaries
importance_gain = model_best.get_booster().get_score(importance_type='total_gain')
importance_cover = model_best.get_booster().get_score(importance_type='total_cover')
importance_weight = model_best.get_booster().get_score(importance_type='weight')

# Convert dictionaries to DataFrames
df_gain = pd.DataFrame(importance_gain.items(), columns=['Feature', 'Total_Gain'])
df_cover = pd.DataFrame(importance_cover.items(), columns=['Feature', 'Total_Cover'])
df_weight = pd.DataFrame(importance_weight.items(), columns=['Feature', 'Weight'])

# Merge DataFrames
df_merged = pd.merge(df_gain, df_cover, on='Feature')
df_merged = pd.merge(df_merged, df_weight, on='Feature').sort_values(by ='Total_Gain', ascending=False )

# Display the merged DataFrame
df_merged.to_csv(os.getcwd() +'/feature_importance.csv', index=False)