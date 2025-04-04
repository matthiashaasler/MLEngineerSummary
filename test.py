import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Sample Data
data = {'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'target': [100, 200, 300, 400, 500]}
df = pd.DataFrame(data)

# Separate features and target
X = df[['numeric1', 'numeric2', 'category']]
y = df['target']

# Define ColumnTransformer
numerical_features = ['numeric1', 'numeric2']
categorical_features = ['category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', Ridge())])

# Define the Parameter Grid
param_grid = {
    'preprocessor__num__with_mean': [True, False],  # Adjust StandardScaler parameters
    'model__alpha': [0.1, 1.0, 10.0],  # Adjust Ridge parameters
    'model__solver': ['auto', 'svd', 'cholesky']  # Adjust Ridge solver
}

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# Run GridSearchCV
grid_search.fit(X, y)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Inspect all results
print("\nAll Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_preprocessor__num__with_mean', 'param_model__alpha', 'param_model__solver', 'mean_test_score']])