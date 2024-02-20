import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

def model_prep_pipleline(X_train,y_train,classifier,num_feat):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, num_feat)])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',  classifier)
    ])

    model = pipe.fit(X_train, y_train)
    return model


def grid_search_param(param_grid,model,X_train,y_train):

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc')

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    all_res = grid_search.cv_results_
    print("All Parameters:", all_res)

    # Make predictions on the test data using the best model
    best_model = grid_search.best_estimator_
    return best_model