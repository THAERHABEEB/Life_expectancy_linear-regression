# ==============================
# Life Expectancy Prediction
# ==============================

# 1. Import libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# 2. Load dataset
df = pd.read_csv("life_expectancy.csv")

# =========================
# Preprocessing
# =========================

# Fill missing values
for i in [" BMI ","Polio","Income composition of resources"]:
    df[i].fillna(df[i].median(), inplace=True)

imputer = KNNImputer()
for i in df.select_dtypes(include="number"):
    df[i] = imputer.fit_transform(df[[i]])

# Outlier treatment
def fu(col):
    q1, q3 = np.percentile(col,[25,75])
    iqr = q3 - q1
    lw = q1 - 1.5*iqr
    uw = q3 + 1.5*iqr
    return lw, uw

for i in ["GDP","Total expenditure"]:
    lw, uw = fu(df[i])
    df[i] = np.where(df[i]<lw, lw, df[i])
    df[i] = np.where(df[i]>uw, uw, df[i])

# Drop duplicates
df = df.drop_duplicates()

# Encoding categorical vars
dummy = pd.get_dummies(df, columns=["Country","Status"], drop_first=True)
print(df.columns)
# =========================
# Features & Target
# =========================
# target column
target = 'Life expectancy '
X = dummy.drop(columns=[target])
y = dummy[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_cols)], remainder='passthrough')

# =========================
# Baseline Model: Linear Regression
# =========================
pipe_lr = Pipeline([('preprocessor', preprocessor),
                    ('model', LinearRegression())])
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)

print("Linear Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
print("="*50)

# =========================
# Compare Multiple Models
# =========================
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

print("Cross-Validation Results:")
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    neg_mse = cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    rmse_scores = np.sqrt(-neg_mse)
    r2_scores = cross_val_score(pipe, X, y, scoring='r2', cv=5, n_jobs=-1)
    print(f"{name}: CV RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f}), CV R2 = {r2_scores.mean():.4f}")
print("="*50)

# =========================
# Hyperparameter Tuning for RandomForest
# =========================
pipe_rf = Pipeline([('preprocessor', preprocessor),
                    ('model', RandomForestRegressor(random_state=42))])

param_dist = {
    'model__n_estimators': [100, 200, 400, 800],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', 0.2, 0.5]
}

rs = RandomizedSearchCV(pipe_rf, param_distributions=param_dist,
                        n_iter=20, scoring='neg_mean_squared_error',
                        cv=5, random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)

print("Best RandomForest Params:", rs.best_params_)
best_model = rs.best_estimator_

# Evaluate tuned model
y_pred_best = best_model.predict(X_test)
print("Tuned RandomForest Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))
print("R2:", r2_score(y_test, y_pred_best))
print("="*50)

# =========================
# Feature Importance
# =========================
rf = best_model.named_steps['model']
feature_names = numeric_cols + [c for c in X.columns if c not in numeric_cols]
fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Top 20 Important Features:")
print(fi.head(20))

# =========================
# Draw top 20 important var
# =========================
plt.figure(figsize=(12,8))
fi.head(20).plot(kind='bar',color='teal')
plt.title("top 20 important variable that effect on life average")
plt.xlabel("Factors")
plt.ylabel("importance of variable")
plt.xticks(rotation=45 , ha='right')
plt.tight_layout()
plt.savefig("importance.png")
plt.show
()

# Save Model
# =========================
joblib.dump(best_model, "life_expectancy_model.joblib")
print("Model saved as life_expectancy_model.joblib")
