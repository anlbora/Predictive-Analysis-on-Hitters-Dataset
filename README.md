# Dataset Story

This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

* AtBat Number of times at bat in 1986
* Hits Number of hits in 1986
* HmRun Number of home runs in 1986
* Runs Number of runs in 1986
* RBI Number of runs batted in in 1986
* Walks Number of walks in 1986
* Years Number of years in the major leagues
* CAtBat Number of times at bat during his career
* CHits Number of hits during his career
* CHmRun Number of home runs during his * career
* CRuns Number of runs during his career
* CRBI Number of runs batted in during his career
* CWalks Number of walks during his career
* League A factor with levels A and N indicating player’s league at the end of 1986
* Division A factor with levels E and W indicating player’s division at the end of 1986
* PutOuts Number of put outs in 1986
* Assists Number of assists in 1986
* Errors Number of errors in 1986
* Salary 1987 annual salary on opening day in thousands of dollars
* NewLeague A factor with levels A and N indicating player’s league at the beginning of 1987

# Import Necessary Libraries

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

!pip install joblib
import joblib

import warnings
warnings.filterwarnings("ignore")
```

# Import Dataset

```
df = pd.read_csv("/kaggle/input/hitters")
df.head()
```
# General Information About to Dataset

```
def check_df(dataframe,head=5):
  print(20*"#", "Head", 20*"#")
  print(dataframe.head(head))
  print(20*"#", "Tail", 20*"#")
  print(dataframe.tail(head))
  print(20*"#", "Shape", 20*"#")
  print(dataframe.shape)
  print(20*"#", "Types", 20*"#")
  print(dataframe.dtypes)
  print(20*"#", "NA", 20*"#")
  print(dataframe.isnull().sum())
  print(20*"#", "Qurtiles", 20*"#")
  print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
```

# Analysis of Categorical and Numerical Variables
```
def grab_col_names(dataframe, cat_th=10, car_th=20):
  #Catgeorical Variable Selection
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category","object","bool"]]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["uint8","int64","float64"]]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category","object"]]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  #Numerical Variable Selection
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8","int64","float64"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  return cat_cols, num_cols, cat_but_car, num_but_cat
```

```
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

#Print Categorical and Numerical Variables
print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f"Cat_cols: {len(cat_cols)}")
print(f"Num_cols: {len(num_cols)}")
print(f"Cat_but_car: {len(cat_but_car)}")
print(f"Num_but_cat: {len(num_but_cat)}"
```
```
def cat_summary(dataframe,col_name,plot=False):
  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                      'Ration': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
  print("##########################################")
  if plot:
    sns.countplot(x=dataframe[col_name],data=dataframe)
    plt.show(block=True)
```
```
def cat_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
  for col in cat_cols:
    cat_summary(dataframe, col, plot=True)
```

`cat_summary_df(df)`

```
def num_summary(dataframe, num_col, plot=False):
  print(50*"#", num_col ,50*"#")
  quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
  print(dataframe[num_col].describe(quantiles).T)

  if plot:
    dataframe[num_col].hist(bins=20)
    plt.xlabel(num_col)
    plt.ylabel(num_col)
    plt.show(block=True)
```
```
def num_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  for col in num_cols:
    num_summary(dataframe, col, plot=True)
```
`num_summary_df(df)`
```
def plot_num_summary(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  plt.figure(figsize=(12,4))
  for index, col in enumerate(num_cols):
    plt.subplot(3,6, index+1)
    plt.tight_layout()
    dataframe[col].hist(bins=20)
    plt.title(col)
```
`plot_num_summary(df)`

# Target Analysis
```
def target_summary_with_cat(dataframe, target, categorical_col):
  print(f"##################### {target} -> {categorical_col} #####################")
  print(pd.DataFrame({"Target Mean": dataframe.groupby(categorical_col)[target].mean()}))
```

```
def target_summary_with_cat_df(dataframe, target):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
    for col in cat_cols:
        target_summary_with_cat(dataframe, target, col)
```
`target_summary_with_cat_df(df, "Salary")`

# Correlation Analysis

```
def high_correlated_cols(dataframe, corr_th = 0.90, plot=False):
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8", "int64", "float64"]]
  corr = dataframe[num_cols].corr()
  corr_matrix = corr.abs()
  upper_triangular_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  drop_list = [col for col in upper_triangular_matrix.columns if any(upper_triangular_matrix[col] > corr_th)]
  if drop_list == []:
    print("Aftre corelation analysis, we dont need to remove variables")

  if plot:
    sns.set(rc={'figure.figsize': (18,13)})
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
    plt.show()

  return drop_list
```
`drop_list = high_correlated_cols(df, plot=True)`

# Missing Value Analysis

```
def missing_value_table(dataframe, na_names=False):
  na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
  n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
  ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
  print(missing_df)
  if na_names:
    print("######### Na Names ###########")
    return na_columns
```
`missing_value_table(df, na_names=True)`
```
def fill_na_median(dataframe):
  dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype not in ["category", "object", "bool"] else x, axis=0)
  return dataframe
```
`df = fill_na_median(df)`

# Encoding & Scaling
```
def one_hot_encoding(dataframe, drop_first=True):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
  return dataframe
```
`df = one_hot_encoding(df)`

# Create a Base Model: Prediction Salary using Random Forest Algorithm

```
X = df.drop(["Salary"], axis=1)
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
```
```
rf_model = RandomForestRegressor(random_state=1).fit(X_train, y_train)
mean_squared_error(y_train, rf_model.predict(X_train))
mean_squared_error(y_test, rf_model.predict(X_test))
np.sqrt(mean_squared_error(y_train, rf_model.predict(X_train)))
np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
cv_results = cross_validate(rf_model, X, y, cv=10, scoring="neg_mean_squared_error")
-cv_results['test_score'].mean()
np.sqrt(-cv_results['test_score'].mean())
```
```
def RF_Model(dataframe, target, test_size=0.20, cv=10, results=False, plot_importance=False, save_results=False):
  X = dataframe.drop(target, axis=1)
  y = dataframe[target]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
  rf_model = RandomForestRegressor(random_state=1).fit(X_train, y_train)
  if results:
    mse_train = mean_squared_error(y_train, rf_model.predict(X_train))
    mse_test = mean_squared_error(y_test, rf_model.predict(X_test))
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, rf_model.predict(X_train))
    mae_test = mean_absolute_error(y_test, rf_model.predict(X_test))
    r2_train = rf_model.score(X_train, y_train)
    r2_test = rf_model.score(X_test, y_test)
    cv_results_mse = cross_validate(rf_model, X, y, cv=cv, scoring="neg_mean_squared_error")
    cv_results_rmse = cross_validate(rf_model, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    print("MSE Train: ", "%.3f" % mse_train)
    print("MSE Test: ", "%.3f" % mse_test)
    print("RMSE Train: ", "%.3f" % rmse_train)
    print("RMSE Test: ", "%.3f" % rmse_test)
    print("MAE Train: ", "%.3f" % mae_train)
    print("MAE Test: ", "%.3f" % mae_test)
    print("R2 Train: ", "%.3f" % r2_train)
    print("R2 Test: ", "%.3f" % r2_test)
    print("Cross Validate MSE: ", "%.3f" % -cv_results_mse['test_score'].mean())
    print("Cross Validate RMSE: ", "%.3f" % -cv_results_rmse['test_score'].mean())

    if plot_importance:
      feature_imp = pd.DataFrame({'Value': rf_model.feature_importances_, 'Feature': X.columns})
      plt.figure(figsize=(8, 6))
      sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
      plt.title("Importance Features")
      plt.tight_layout()
      plt.savefig("importance.jpg")
      plt.show()

    if save_results:
      joblib.dump(rf_model, "rf_model.pkl")
```
`RF_Model(df, "Salary", results=True, plot_importance=True, save_results=True)`
```
def load_model(pklfile):
  model_disc = joblib.load(pklfile)
  return model_disc
```
```
X = [300, 70, 1, 40, 50, 20, 2, 200, 70, 1, 40, 40, 20, 500, 40, 30, True, False, True]
model_disc = load_model("rf_model.pkl")
model_disc.predict(pd.DataFrame(X).T)[0]
X = df.drop("Salary", axis=1)
random_baseballer = X.sample(1, random_state=1).values.tolist()[0]
model_disc.predict(pd.DataFrame(random_baseballer).T)[0]
```


















