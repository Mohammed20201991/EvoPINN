! pip install statsmodels

## Imports and data loading
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_excel("/content/drive/MyDrive/RegressionDiagnostics/Suction_vsCP-modified_1.xlsx")
data.head(10)

# Define the mapping
original_columns = list(data.columns)
new_columns = {}

# Loop through and rename all except the target column
x_counter = 1
for col in original_columns:
    if col.strip().lower() == "collapse potential (%)":
        new_columns[col] = "Y"
    else:
        new_columns[col] = f"X{x_counter}"
        x_counter += 1

# Rename the columns
data.rename(columns=new_columns, inplace=True)

# Display the renamed columns and first few rows
print(data.columns)
print(data.head())

"""## a)

We start by using the simple linear regression model
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3+ \beta_4 X_4 + \beta_5 X_5 + \beta_6 X_6
$$

Using [``sm.OLS``](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html), To compute optimal values for the parameters.

> Note: We use ``sm.add_constant`` to add values for the intercept.


"""

# Prepare input data
X = sm.add_constant(data[["X1", "X2", "X3","X4","X5","X6"]])
y = data["Y"]

X[:3]

X.shape

y[:3]

y.shape

# Fit a linear model with statsmodels
model   = sm.OLS(y, X)
results = model.fit()

model.__dict__

# Show the results using the summary() function
print(results.summary())

"""Use the model to predict the $y$-values"""

predicted_values = model.predict(results.params, X)

predicted_values[:2]

# Visualization of the predicted variables vs. the true variables
fig, axs = plt.subplots(1, 6, figsize=(20, 5))
for ax, variable_name in zip(axs, ["X1", "X2", "X3","X4","X5","X6"]):
    ax.scatter(data[variable_name], data["Y"], label="Ground truth")
    ax.scatter(data[variable_name], predicted_values, label="Model prediction")
    ax.legend()
    ax.set_xlabel(variable_name)

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()  # Flatten to easily iterate over

# List of feature names
features = ["X1", "X2", "X3", "X4", "X5", "X6"]

# Plot for each feature
for ax, feature in zip(axs, features):
    ax.scatter(data[feature], data["Y"], label="Ground truth", alpha=0.7)
    ax.scatter(data[feature], predicted_values, label="Model prediction", alpha=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel("Y")
    ax.set_title(f"{feature} vs Y")
    ax.legend()

plt.tight_layout()
plt.show()

"""## b)

Compute the residuals $e = \hat{y} - y$ of the resulting model.
"""

residuals = data["Y"] - predicted_values
# alternatively: residuals = results.resid

residuals[:4]

"""Plot the residuals over the input variables $x_1$ and $x_2$. What do you observe?"""

plt.figure()
plt.scatter(data["X1"], residuals)
plt.figure()
plt.scatter(data["X2"], residuals)
plt.figure()
plt.scatter(data["X3"], residuals)
plt.figure()
plt.scatter(data["X4"], residuals)
plt.figure()
plt.scatter(data["X5"], residuals)
plt.figure()
plt.scatter(data["X6"], residuals)

# Residual plot layout
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

# Feature names
features = ["X1", "X2", "X3", "X4", "X5", "X6"]

# Plot residuals vs each input variable
for ax, feature in zip(axs, features):
    ax.scatter(data[feature], residuals, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs {feature}")

plt.tight_layout()
plt.show()

"""Using a White test ([`statsmodels.stats.diagnostic.het_white`](https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_white.html)), show that we can reject the hypothesis of homoscedastic residuals at an $\alpha$ level of 0.01."""

from statsmodels.stats.diagnostic import het_white

statistic, p_value, _, _ = het_white(residuals, X)
print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
print(f"p-value of the statistic: {p_value}")

"""## c)

Consider the alternative model
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4  + \beta_5 X_5 +  \beta_6 X_6 + \beta_4 X_1^2
$$

Compute the optimal parameter values. You should observe that the $R^2$ value improves drastically over the previous model.
"""

# Prepare input data
X = sm.add_constant(data[["X1", "X2", "X3","X4","X5","X6"]])
X["X1^2"] = np.square(X["X1"])
y = data["Y"]

# Fit a linear model
model   = sm.OLS(y, X)
results = model.fit()

print(results.summary())

X[:10]

"""Although this model gives a very good fit of the data, there is another problem.
Use the Variance inflation factor ([`statsmodels.stats.outliers_influence.variance_inflation_factor`](https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)) to check whether the variables are dependent.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

for index, variable_name in enumerate(X.columns):
    if variable_name == "const":
        continue
    print(f"VIF for variable {variable_name} is {vif(X, index)}")

# Bonus: Check if residuals are now homoscedastic
from statsmodels.stats.diagnostic import het_white
original_X_for_white_test = sm.add_constant(data[["X1", "X2", "X3","X4","X5","X6"]])


statistic, p_value, _, _ = het_white(results.resid, original_X_for_white_test)
print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
print(f"p-value of the statistic: {p_value}")

# statistic, p_value, _, _ = het_white(results.resid, X)
# print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
# print(f"p-value of the statistic: {p_value}")

"""## d)
Consider a third model:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_4 X_4 + \beta_5 X_5 + \beta_6 X_6 + \beta_3 X_1^2
$$

Compute the optimal parameter values.
"""

# Prepare input data
X = sm.add_constant(data[["X1", "X2","X3", "X4","X5", "X6"]])
X["X1^2"] = np.square(X["X1"])
y = data["Y"]

# Fit a linear model
model   = sm.OLS(y, X)
results = model.fit()

print(results.summary())

X['X1'][0] ,X["X1^2"][0]

"""Check if the model has multicollinear input variables using the VIF."""

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

for index, variable_name in enumerate(X.columns):
    if variable_name == "const":
        continue
    print(f"VIF for variable {variable_name} is {vif(X, index)}")

"""Check if the model satisfies the homoscedasticity assumption using the White test and an $\alpha$ level of 0.01."""

from statsmodels.stats.diagnostic import het_white

# statistic, p_value, _, _ = het_white(results.resid, X)
# print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
# print(f"p-value of the statistic: {p_value}")

# Create the exogenous variables for the White test using the original untransformed variables
X_for_white_test = sm.add_constant(data[["X1", "X2"]])

statistic, p_value, _, _ = het_white(results.resid, X_for_white_test)
print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
print(f"p-value of the statistic: {p_value}")

# Bonus: Visualization of the residuals
plt.figure()
plt.scatter(data["X1"], results.resid)
plt.figure()
plt.scatter(data["X2"], results.resid)
plt.figure()
plt.scatter(data["X3"], results.resid)
plt.figure()
plt.scatter(data["X4"], results.resid)
plt.figure()
plt.scatter(data["X5"], results.resid)
plt.figure()
plt.scatter(data["X6"], results.resid)

# Visualization of the predicted variables vs. the true variables
fig, axs = plt.subplots(1, 6, figsize=(20, 5))
for ax, variable_name in zip(axs, ["X1", "X2", "X3","X4", "X5", "X6"]):
    ax.scatter(data[variable_name], data["Y"], label="Ground truth")
    ax.scatter(data[variable_name], model.predict(results.params, X), label="Model prediction")
    ax.legend()
    ax.set_xlabel(variable_name)

"""# Regression Pipeline:"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_excel("/content/drive/MyDrive/RegressionDiagnostics/Suction_vsCP-modified_1.xlsx")

# Rename columns for regression
data.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y']

# Add polynomial term manually
data['X1_squared'] = data['X1']**2

# Define predictors and target
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X1_squared']]
y = data['Y']

# Add intercept for statsmodels
X_sm = sm.add_constant(X)

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def remove_high_vif_features(X, threshold=10.0):
    while True:
        vif_df = calculate_vif(X)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            drop_feat = vif_df.sort_values('VIF', ascending=False).iloc[0]['feature']
            print(f"Dropping {drop_feat} due to VIF = {max_vif:.2f}")
            X = X.drop(columns=[drop_feat])
        else:
            break
    return X

# Remove features with VIF > 10
X_noconst = X_sm.drop(columns=['const'])  # exclude intercept before VIF
X_reduced = remove_high_vif_features(X_noconst)

# Add intercept back
X_reduced = sm.add_constant(X_reduced)

ols_model = sm.OLS(y, X_reduced)
robust_result = ols_model.fit(cov_type='HC3')  # Robust to heteroscedasticity
print(robust_result.summary())

# Use reduced set of features from VIF filtering (without intercept)
X_ml = X_reduced.drop(columns='const')
X_train, X_test, y_train, y_test = train_test_split(X_ml, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Lasso Regression
lasso = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Evaluation
def eval_model(name, y_true, y_pred):
    print(f"---- {name} ----")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print()

eval_model("Ridge", y_test, y_pred_ridge)
eval_model("Lasso", y_test, y_pred_lasso)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Y', marker='o', linestyle='', alpha=0.5)
plt.plot(y_pred_ridge, label='Ridge Prediction', linestyle='--')
plt.plot(y_pred_lasso, label='Lasso Prediction', linestyle='-.')
plt.title("Ridge vs Lasso Predictions vs Ground Truth")
plt.xlabel("Test Sample Index")
plt.ylabel("Collapse Potential (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame({
    'Feature': X_ml.columns,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
})

# Add OLS robust model coefficients
coef_df['OLS (Robust)'] = robust_result.params[X_ml.columns].values

# Plotting
coef_df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
plt.title("Coefficient Comparison: Ridge vs Lasso vs OLS")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()

"""# 3-"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === Load Data ===
data = pd.read_excel("/content/drive/MyDrive/RegressionDiagnostics/Suction_vsCP-modified_1.xlsx")
data.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y']
data['X1_squared'] = data['X1'] ** 2

X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X1_squared']]
y = data['Y']

# === VIF Filtering Function ===
def calculate_vif(X_df):
    vif = pd.DataFrame()
    vif["feature"] = X_df.columns
    vif["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    return vif

def remove_high_vif_features(X_df, threshold=10.0):
    while True:
        vif_df = calculate_vif(X_df)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            drop_feat = vif_df.sort_values('VIF', ascending=False).iloc[0]['feature']
            print(f"Dropping '{drop_feat}' with VIF = {max_vif:.2f}")
            X_df = X_df.drop(columns=[drop_feat])
        else:
            break
    return X_df

# === Step 1: Remove multicollinear features ===
X_filtered = remove_high_vif_features(X)

# === Step 2: Train-Test Split and Standardization ===
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 3: Ridge and Lasso Regression ===
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

lasso = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# === Step 4: Robust OLS Regression ===
X_sm_robust = sm.add_constant(scaler.transform(X_filtered))  # All rows for statsmodels
ols_robust = sm.OLS(y, X_sm_robust).fit(cov_type='HC3')
print(ols_robust.summary())

# === Step 5: Evaluation Function ===
def eval_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))

eval_model("Ridge", y_test, y_pred_ridge)
eval_model("Lasso", y_test, y_pred_lasso)

# === Step 6: Plot Prediction Comparison ===
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Y', marker='o', linestyle='', alpha=0.5)
plt.plot(y_pred_ridge, label='Ridge Prediction', linestyle='--')
plt.plot(y_pred_lasso, label='Lasso Prediction', linestyle='-.')
plt.title("Ridge vs Lasso Predictions")
plt.xlabel("Test Sample Index")
plt.ylabel("Collapse Potential (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ridge_lasso_predictions.png")  # Optional save
plt.show()

# === Step 7: Coefficient Comparison ===
coef_df = pd.DataFrame({
    'Feature': X_filtered.columns,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
})

# OLS Robust coefficients
coef_df['OLS (Robust)'] = ols_robust.params[1:len(X_filtered.columns)+1].values  # exclude intercept

coef_df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
plt.title("Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_coefficients.png")
plt.show()

"""![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABHsAAAERCAYAAADrBqt1AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAHgmSURBVHhe7d13fE33/wfw182qG+7NIktEiETtEBUUFWrXqFHUSrWltUqtTn5GdVBUKdEapaqqVGrVjrZmJbFDbAlZgkSW3PH5/SH3fHPOvYmEGL1ez8fjfL/1GSc3N+d+7ue8z2eohBACRERERERERERkFWyUCURERERERERE9N/FYA8RERERERERkRVhsIeIiIiIiIiIyIow2ENEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWRCWEEMrE0pCZnokbN25DqIyy9Lu5dwGVCvZ2djAaBOwcbGX5BeVk58BgECincfxfogBsoEIFrwpwdCxTsDgRERERERER0TOv1IM98ZeTcScjA2XUzymzAACpySlQ2ahQvkIFZZaZxOtJyM3NQ5Wqvv9LVAG2KhvY29nDxtYGz6nVcHErV7AaEREREREREdEzq1SncaUm3ULStWRA2MCQB4uHEPcOZbqlA+LeUTDNeFfAqFPBoAdUUMGo0yMrPUf5UoiIiIiIiIiInkmlGuy5eO4KhFHAaCj8AO4Fe5Tplg4h7sV7CqYZTP+tN8KgF9DrDUi/naV8KUREREREREREz6RSDfbk3b2rTDJX0kljAoi7qccn/+ow8bAOEw/r8f7Bu5gUo4dBZ4ReZ4AwGpS1qAhJSUmoWbMmtFqtdHTs2BFZWaUbNDt69Cg8PDxkP2fo0KHKYs+cO3fu4Ntvv0WTJk3g7OwsvTctW7ZEamqqsjgRWTEhBH744QdUqlQJWq0WrVu3xvnz55XFiIiIiIhKpFSDPSUO5BTTrKPZuKoqh0QHVyQ6uCL1OVdcVmkx7bgeRoMRpRHrsRQAKXj89ttvyioW/frrr2Z1TcejCKg8CywFjSwdvr6+6NKlC9asWYPs7GzlaZ4Khw8fRt26dfHxxx/j1KlTMBr/t4B5SkoKDIZSuJjpmSSEwPnz5zFp0iQEBwfDzc1N9vmoWrUqWrVqhRkzZuD69evK6vSEREZGYty4cUhPTwcA/Pvvvxg/fjxycjg9mYiIiIgeXOkGe4pDpYJKmVaY/IKGPANgay8dds89hzKOauj0AIQKpbzGtEVbtmyBXq9XJsvo9Xps3LhRmUyPye3btxEZGYm3334bAQEB+PPPPx/LtVFcFy9eRP/+/ZGWlqbMInpgQgjs3bsXL7zwAho0aIC5c+fi3Llz0Ol0snI3btzAkSNH8MUXX+DYsWOyPHpyjh49Kgv6AsDZs2el4A8RERER0YN47MEeFVRQiWL+WKMK+jzjvUV+ClCpABsbFexghEqlgngMv8b+/ftx7do1ZbJMQkICDh48qEymJ+DOnTvo06cP/vjjD2XWE7NmzRokJSUpk1G/fn10794dXl5eyiyiIul0OsycOROdO3dGXFycMpv+Axo2bAgbG/l3WPXq1eHk5CRLIyIiIiIqiVKPkhiFwCHb7YhUf429Zb/E3rJf4p9yXyHG/RtcLrsNUNlAZVO8sT1CAHl5Oqggf+pZkEplA0U/+ZG4fv06/v33X2WyzOHDh5GcnKxMpkfA3t4ePj4+8PHxgbe3t9nNEgAYjUZMmzYNKSkpyqzHLjs7GwcOHJClOTg4YOPGjdi7dy+WL1+OXbt2wdPTU1aGqDA6nQ4TJkzA9OnTlVn0H9KsWTPMmjVLCu40bdoU33zzDdRqtbIoEREREVGxmd8hP4TcvLtIT0/HNWM8tA65cHXIu3c8lwcPjQ4odw4QgBAq5OXdve9hFAJ38/QQhcR6jMKI7Nws6PLylFkPTaVSwd/fX5ZW1FSu3Nxc/Prrr9K/3dzc4O3tLStDpadnz544ffo0Tp8+jTNnzuD69euYMGGCshji4uJw5MgRZfJjJ4Qwu3bc3d1RvXp1WRpRcQghsGDBAixZskSZBQBo3rw5vv/+e5w6dQpxcXGIi4vD5s2b8fXXX6NFixZwcHBQVqEnRKVS4a233kJ8fDwyMjLw559/wtfXV1mMiIiIiKhESjXYc/rMKVy4dA62RsvrpAidHXJzcpCTnY2bN9Pue9xJz0TajXQY9AbAaCxwCEAI5OXl4Xb6LdzJzFD+qIcmhED9+vVlaX///TcSEhJkaSZXrlyRjfzx9fXlYruPkaOjI8aOHYvWrVsrs7Bnzx5lEtF/2pkzZ/DNN98ok+Hj44PIyEhs3rwZvXv3RqVKleDp6QlPT080b94cb7/9NjZt2mTxc0JERERERNajVIM9Deo2QPXA2rCxtVdmAQDs7Ozg6FgWZcuWg6en930PJ2cttE7lYGsDqIQRKmP+Ie6t46N+7jn4VPSFu7u78keVioCAAJQrV076d3JyMg4fPiwrY7Jv3z7cunVL+vcLL7wAe3vL70NhsrOzsWbNGrRq1Qqenp7SLjpubm5o1aoVFi5ciDt37iirmRFC4MiRI+jVq5d0HmdnZ7Rt2xb79u0zWwz0fgwGA7Zt24ZXX31VtsNP9erVMXbsWFy5ckVZ5YlQq9Vo0aKFMhkZGebBQNP25yEhIdLv4+zsjCZNmmDx4sWF7uY1Z84cs13Atm7diuzsbEyePFnaPnnEiBGy8l5eXvjnn39k50pISEBgYKB0njlz5sjykf/eHzp0CG+88QaqVq0q+7n16tXDhAkTcO7cOWU1iaWdzIYOHQohBCIiIhAcHAytVou6detK092GDh0qK+/h4YGjR49CCIF9+/ahbdu20pbx/v7+mDRpkuy6vHPnDiZNmgR/f3/pHLVq1cKqVavMFg0u6ObNm1ixYgX69u0r+11Nf5fVq1cXukNRYb8nAFy7dg0jR460+Fm43wLepmt/wIAB8PX1lZ2/Zs2a+OijjwpdcPtBr7H7EUJgyZIlZj/Xz88PGzduRIMGDWTpJfGorjdYuC4s/R0sXWNubm4YOHAgLl26pPhp9xR2vRoMBqxfv172/vv7+2Ps2LFFrr9mMBhw5MgRfPjhh3jxxRdlbZ6vry/efffdIt+Dwl5PcnIyRo4cKZ1v/vz5wH3eM6WLFy9i0qRJqFevnqy8r68vWrVqhZUrV5qNIDR52O+Xwto+S38zT09PjBw5ssj3mYiIiIgerVIN9hQkoJId/0srPmH6H6PIP+QjeyAAGxVgY2OrrFoqqlWrhqCgIFnaxo0bzTrTubm52LJli/RvtVpdoifnQgj8+eefCAgIwNtvv40jR47IbgR1Oh2OHDmCiRMnolq1avjtt98KvUnV6XT45JNP0KpVK2zbtk06j9FoxMGDB9GhQwcsWLBAWa1QV69eRbt27dCrVy/s2rVLdrOemJiI77//HvXq1cPChQv/EyOZTO/1888/j48//hixsbFSntFoxKlTpzBu3Dg0bNgQ0dHRsrqFyczMRN++fTFnzhxpB52ighrFZXrv27Rpg3Xr1uHGjRuy/EuXLmHRokUIDg7G2LFjSxQ8mD9/PgYMGCDdtOr1+iKDgJmZmRg2bBg6dOiAgwcPSmVTU1Mxd+5c9OvXDxkZGTh+/DhCQkIwd+5cpKamSvXj4+Px7rvvYuzYsRbfm0WLFsHPzw8jRozA5s2bZb+r6e8ydOhQNG/eHGfPnpXVLcqePXvQuHFj/PjjjxY/C/Pnzy/0sxQdHY0GDRqgV69eiIiIwO3bt2X5CQkJmD9/vlkA+FFcYwWlpqYiMjJSmYyPP/7YbOppSTzK623v3r2oW7eu7Low/R06deqEP/74Q2q7lNeYTqfDhg0b0KJFC0RFRSnObNnt27fRq1cvhIWFyd7/1NRUfP/992jQoAG2bdsmq2PKb9iwIVq1aoUFCxbgxIkTsuv19u3bWLVqFYKDg/HVV19ZvJYtSUhIQKdOnfDjjz9KdYpbF/llJ02ahKCgIMydO9cs8HX79m0cOXIEq1evxt27d2V5pfn9opSXl4cvvvjC7G+WnZ2NH3/8Ec2aNUNMTIyyGhERERE9Bo8k2CMAGGADPWylwwgbGGFzL7N4/cj/lTMKqAocpkCPKdpT3AWfSyo5ORlt2rSRpR08eNBsKtfFixdx6NAh6d/BwcEIDAyUlSmMEALz58/Ha6+9VuRTVZOcnBwMHjwY33zzjVmHXK/XY8KECfj2229l6Urz5883+x0suXDhAl555RWzm1klo9GIiRMnYvny5cqsx87Se6jVaqX/XrNmDfr06WOxXEEJCQno0aOH7EaxMBs2bCj1qWIxMTF46aWX7vvem3z//ffo06ePxVFMSmfPni1RwA8APvroI6xatUqZLImMjMSYMWMwaNCgIq+tH3/8EVu3blUmFzpiRykuLg7vvfdesX7PqKgoDB48uMgtrGfPno0zZ84ok7Ft2zZ06NDB7Ka6OB7FNVbQhQsXcPHiRVlaYGAgQkNDZWkl8Sivt6ioKISFhZmNRDIxGo0YP3483n///SLbrvT0dHz66afIyspSZsnodDqMHz8eO3fuVGZJcnJy8M4775i99waDwSxYUpjPPvus2Lv9LV68+KF2S1u0aBHmzp2rTL6v0vx+sSQ8PByff/65MlmSlpaGzz//vNifbyIiIiIqPY8k2GOEShboMcAGAioY80f4lJhsVE/+cW+lZ9iIe1uxPwo6nQ7t2rWTbYGbnJyM3bt3y8r9888/shvKNm3aoGzZsrIyhYmKirK4m469vT06deqETp06wdHRUZmN6dOnm+3u9Pfff2PZsmWyNOSvZ9OpUye0aNECNjY2xerEZ2Vl4f3338fly5elNHt7e3z00Uc4deoUduzYgbZt28rqzJw584lO6UpNTcXmzZuVyQgJCQEAHDt2DOPHj5eNYKlSpQp++uknxMbGYunSpfDx8ZHy0tLSMHPmTLORXEobN25UJklq166NsLAw9OvXD25ubrI8BwcHafRBWFgYateuDeRPO/vwww8t3hzXr18fffv2RaVKlZRZiIyMxLx58+77942Ojsb169eVyYXKycnB0aNHpWtSuZaVydq1a3HhwgWpXN26dZVFIISQpn5YUr58eYwePRqbN29GXFwcIiIi0LFjR1mZ/fv3Y/v27bI0S+Li4pCWlgYvL69Ct7ZPS0vDrl27ZGkXL17EyJEjLd6gms7VsmVLi9M0H9U1VlB8fLxZ+Ro1asDV1VWWVlyP+noz/R0CAgLQpUsXi1uKJyUl4ccffwTyp8926dLF4m5UUVFRRU6hQn7Q2zT6KyAgAL169bL4+tPS0rBkyZJCX39oaCgWLVqEmJgYxMTE4KuvvoJGo5HyRf4i2fcLeOXk5FgciVVcaWlpWLlypSxNrVZj0qRJ0mtbsWIF3nnnHbMpzaX5/WKJaWpq/fr10alTJ4ufiX/++Qfnz59XJhMRERHRI/ZIgj1CMarn3sge1f9G9hRXgZE9skCPKDi6RwVb1SP5NYD8G7WGDRvK0rZs2YLc3FwgvyNfcKSCWq0u9hN2vV6P7777zuymcuDAgUhISMDq1auxevVqXLp0CR988IGsTF5eHn744Qfppk+v12PlypVmU3FMa12sXr0amzZtQnR0dLGmemzdulU2WsXGxgZLly7FBx98gEqVKiEkJATfffedbATT9evXzW6cHwedToeDBw/i1VdfNXtSHxgYiBYtWkCv1+Obb76RBeX8/f2xadMmdOnSBRUrVkTPnj0xb948qApED3ft2oULFy5I/y5M06ZNcfToUaSnpyMlJQUTJ04E8gN/8+bNw6xZs1CjRg1ZHXd3d3z22WeYN28e5s2bJ40i2759O/bv3y8r6+/vj6NHj2Lv3r0IDw/HyZMn8fvvv5vdOP/000+4evWqLM0SJycnrF27Fmlpabh16xb+/PNPs3MVZPr5q1evxt69e7Fs2TLZ+2Sp3N9//42FCxcqi+D06dNmo218fX0RERGBc+fOYerUqWjevDk8PT0RGhqKJUuWoFmzZrLyxb15njx5Mk6fPo3ly5fj5MmT6Nevn7IITpw4If23EALfffcdkpKSZGUCAwOxf/9+nD17FsuXL8cff/yBpKQkzJ8/H2XKlAHyP4OP8hozsbQOiqOjI+zs7JTJxfI4rrepU6fiyJEj+Omnn3Dq1Ck0bdpUWQSwUE4ZWMzJySnWCBknJydEREQgKioKS5YswcmTJ7Fs2TLY2Mi/K7Zu3YrExETp37a2tnj77bdx+fJlRERE4PXXX4e/vz/8/f3xzjvvYOHChbK/3ZkzZ4od4O7VqxfOnz+PjIwMJCQk4LXXXlMWsSg+Pt7sZ3Tr1g3jxo2TXlu3bt3w1VdfYfny5dKDhtL8fimMk5MTduzYgb1792L16tWIiYmBn5+frExmZmaRo/2IiIiI6NF4JFESAZgFe+6N7Ln34wo+SL17IBZ5e09Ih9j7vxuv/zECBRdoNhqhEuLeQs0AhPk9Z6lRq9Xo3LmzLO3ff/+VOt+XLl2Sbe3dtGnTYm+nnZCQgL///luWFhgYiEmTJsmeaqvVarzzzjuoV6+erGzB3cGKe66qVati5syZFm/UTfR6vWwNIuSPjlGuQ+Tu7o7g4GBZ2oOsQfIgVq9eLVtgtG3btjh+/LisjI2NDT799FO4u7vj2rVrZje0/fr1M3viX7t2bVSsWFH6961bt+57M+vn54fFixejatWqUKlUKFOmDKpWraosVix6vR4RERGyNJVKhZkzZ8rOqVKp0KpVK7zxxhuystevX5dNKbREpVJh7ty5aNeuHezt7WFrawtfX1+LIylM3n33Xdl71bp1a7P1rJC/boypnEqlQmhoqNl7nJqaKgVLTXr06IHQ0FDY2srX39LpdMjKykL58uVl6RcvXrzvdJ7AwEAMGDBAOqe9vT0GDhxoFhSJj4+XzpWYmGg2zczNzQ0rV66URl6ZmM5nCu4+ymvsUXkc11tgYCBef/11qc3RarUYPHiwshgaNmyIwYMHS+XKly+Pbt26KYtZDHYpvf/++7Kgu0qlQteuXc3a8sTERFkgpUKFChgzZozFUVLp6ekoW7asFNxDCQIZTZs2xZw5c6SRN1qtVnYNFMXW1tbsc7Flyxbs2LGjyHXSivudUJzvl8L07t0bjRo1kv7t6+uLnj17ysogPyhGRERERI9X6QZ78uMHAioYZNO47o3sKdgtzc7JRer/LYXT0sNw/Snm3vHzUVTYdhb2cxVTcUyLMpsO07o/AjAiP+0RevHFF+Hi4iL9+9atW9i3bx9gYQpXhw4dirxpLujSpUvSDkgmoaGhZkPxAcDV1RWNGzeWpaWlpUlPpRMTE82mYRR2LuXNplJmZqbZSIMDBw7A29vbbDeW1atXy8olJSWZ3cg/KdOmTUOXLl2A/JtS5Xs9depUs98nMDDQ7AYnPj5e9m+lnj17wtfXV5n8QNLT080WIK5evbrZCAcUCKYoA3f3u7EKCgoyC9wVRa1Wm41uK1eunNnvrFarUa1aNVmak5MTKleuLEsrTHZ2Nn7//Xe8+eabqF27NpydneHm5oZq1aphw4YNyuL3FRwcbHb9V6lSBZ6enrK0gq5cuSIb6QEA7du3x/PPPy9Ls+RRXmP3o9frC52OVJTHcb3Vq1cPFSpUkKVVqlTJLOgWEBAAbYH1tQAU631XUqvVeOmll5TJsLOzM9upTK/Xmy28LYTA+fPnMWPGDLRq1Uraha1SpUp49dVXzUbKFMfgwYPNfrfiqly5stn7kJ6ejh49esDPzw8TJ040W8MJpfz9UpjWrVubXQ8vvPCC7N9ERERE9GSUarBHJe26BRiECgZhk3+oYDACBqikqVkZGdkwZGXDXtjCXtjBXtjBQdjBQWUPm8y8e4VMQR2DKdhjYSoXAKhKfpNTEv7+/mZbem/ZsgW3b9+WjQJwcnIym25SlNzcXLMbNG9vb9m/C1LmFbxRuX37ttmQe2X54srNzZXtpFQS2dnZRT5tfhzq1q2L7du3Y+TIkdKNiKX3p7iUU46USvPmRqfTITMzU5ZWvnx52WiCglxcXMzy7jfy4fnnn4ezs7MyuUjK6S92dnYW1/p4EEIIrFu3DgEBARg0aBDWrl2Lq1evmk1JfBwsXSeBgYFmN7SWWKpbXPe7xgqyFKg9f/58ic5h8jiuNzs7O7P3T61WW1zfpTS4uLjAw8NDmVwsycnJ6N+/Pxo0aIAvvvgCR44cMQsGlZRarS72gv2WaLVavPfee2afQeRfNwsXLkRQUBD69euH5ORkKa80v1+IiIiI6L/HvPdYCoTq3ro9psMIGxiNKghhaev1AsN0FLkFU5VTuADTuj0Cjzq2YGdnZzb8/9ixYzhw4IBs2lLDhg1RpUoVWTl6NOzt7eHj44OAgAD07dsXYWFhmDt3Lk6ePIm///7b7Ck1Pb22bt2KN99802y3IB8fH3Tv3h3vvfdeqQbU/ussjYo5evSobDrps8zGxsZs2lNxZGVlYciQIWYLrjs6OqJhw4YICwvDsGHDzN77x6Fr167YtGlTkd8vGzduRKdOne4bfCMiIiKiZ0OpBntupt9GVmY6hBDQQyUdBqigEyrkCQGD3gC9zgC9TgchjBAwyA5AQAgj7ubehT5PD6PeCJXBAAjFblwCMOj1uJORiez7rN1RGkJCQmRPPlNSUrBlyxbcunVLSivJFC7kB5GUlDe8BSmnVqnVaos7DJkUdq709PT7rnei1KxZMyQmJiIjI6PIY8uWLcXeiexh9OzZE6dPn0ZUVBTCw8Mxb948DB48GL6+vmajCAozZcoUs9dv6RgzZoyy6iNjY2Njdl1kZWVBp9PJ0kxSU1PNppUop3w8zbKysjB//nzZKJ4WLVrgypUr0sLK06ZNM5se9jiVZOcypUdxjQUGBpq9H0IIjB8/vsTTwazxert9+3ahIxOVf0tV/hpbALB3716zRemXLVuGxMRE7N69G/PmzUOfPn0e2Yik+2nWrBmio6OxY8cO9OjRw+LriIuLw6+//go8hu8XIiIiInq6lWqw59LlC4hPiIfeKGBU2cGQf+hhB72wQZ7BiNzsXORk5SA9JRVGowFCVfC4t6W6wWjAjeQ0ZGVmIzMrG0ajweIULt3dPNy8efOxDDWvWLGibAcZIQRWrFgh/dvFxcXiOhFF8fX1la0FhPxdhjIsbOWbkpKCgwcPytI8PDykAJSHh4dZoKmwc+3du1cWpFLSarVmN5MnTpww2xHmv8THxwflypWTpf31119mN65PmmlNl4JOnTplcQciIQS2bdumTDZbSPhplpqaaraV9pAhQ2SfC71ej+zsbFmZR6Vq1apmn8mIiAiLa6IoPa5rzM3NDa+++qoyGRcuXED//v2LXOzZYDDgt99+kxaStsbrLTMzE6dPn1YmIyUlRRbMQf4i86bRMsqRUSEhIWjXrp0seJyTk1NoIOxxsLW1RUhICJYtW4akpCTMmzfPbHqXaQ2l0vx+ISIiIqL/nlIN9gTXC8bzz9eGyta2wHo999bs0RtVgK0dymrKQqMtB7eK3rCxtTEb2SMgYGtri4qVveHkooWTthxs8td+Nh2m9XscyjyHipUqwquIBVdLi52dndkuIwXXQ3jhhReKvRCtSeXKlc2mpxw5cgTz58+X3SDm5ORg6tSpZjdgbdu2lRY+rVy5stnPP3LkCL7//nvZGjqHDh3CtGnTZOWUHB0d0aRJE1laeno6Zs6cafGmWwiBPXv2YOXKlcqsp4alRU53796NtWvXmq1rgfz3fMGCBbJtuR+HMmXKoGPHjrK0vLw8TJo0STY9QwiB9evX44cffpCVDQwMNNtV52l2+/Zts5tP5bbnmzdvNpta86j4+vqaLeKblJSEN954wyyIkpOTg7lz50oBhMd5jQ0cONAsSAMAMTExqFu3LsLCwvD7778jPj4eSUlJ+PvvvzFp0iQEBARg8ODB0vo+1nq9ffbZZ7KFpwtrQxs3bgwfHx/AwtpDaWlpsvYuIyMDn3/++QOvy/SgTpw4gQULFpgFDe3t7dGzZ0+EhITI0k1K8/uFiIiIiP57SjXYYyKggh42BbZet0OesIXOtPV6gZICetlxL/d/UzoE/rcO8/8IGI2Awai6N7vLLP/RCAoKMttG2aRjx45mC5feT5kyZTB8+HCzJ7NffPEFqlSpgr59+6Jv376oUqWKbBQR8p/uv/nmm9JT58Ke9k+ZMgWBgYEICwvDSy+9hDZt2hRrIdfevXub7Vq0bt061K5dG5MmTcKGDRuwYcMGTJ06FUFBQejatStu3LghK/800Wq1GDp0qCxNCIERI0agefPm+P7777Fhwwb8+uuveOedd+Dn54epU6c+kcWmO3fujBo1asjS9u/fj1q1aiE0NBRDhw5F7dq18cYbb5gtYvzuu+9a3G3naVWhQgWz0Qcffvghxo8fjw0bNmD8+PEYPHiw2e/5qKjVaowaNcrsMxkTE4PatWujXr16GDp0KFq1agUfHx9MmjRJ2n3ucV5jFStWRHh4OJycnJRZMBqNWL9+PQYNGoRatWohMDAQnTp1wty5cy1+Rq3xert8+TJCQkIQGhpaaBtqY2ODwYMHS9OdAgICZPlxcXHo3r07VqxYgRUrVqBjx46IjIyUlXkcDAYDpk6dKu289ffffyMpKQl79uzBqFGjcODAAVn54OBgoJS/X4iIiIjov+eRBHuMQIFAjy10sEWesIX+3srNUrRHQEBAvm6PKTW/QP7/y7dbF0bAqDfCIFQwGu+d4XHw8vJChw4dlMlwcXHBiy++qEwulpYtW+L//u//lMnIzs7G5s2bsXnzZrPRNDY2Npg7d67ZDVphT/tTU1Oxfv16xMTEAPnbTyunfClVrVoVM2fONLtRuHHjBubOnYuBAwdi4MCBmDVrFi5duiQr87Tq3r07wsLClMk4fvw4xo4di4EDB+Ktt97Czz//bPYU/XGqUKECvvvuO7MbeaPRiKioKKxevdri2ixhYWEYOHCgMvmp5uHhIZseifwdosLDwzFw4ECEh4fDYDDc93otTYV9JpG/nfXq1atx5MgRi9N5Huc1FhwcjB9++OGh3xtrvN5UKpX0+i21ochvLwvuntiiRQs4ODjIyhw/fhwjRozAiBEjcPz4cTg6Oj6xAEhOTg4WLlyITp06ITAwEF27dsVvv/0mK2MK7JkUdi0/yPcLEREREf23PJJgj1CpYIQtDKZD2CLPaAudUHSSVUKxZs+9YI8ZY36EJ/8QRgOEQZ//lLnAFuyPmEqlQvv27c06+y1atIC/v78srbhUKhXee+89hIeHW1xwU8nNzQ0RERHo2rWrMgsVK1bE0qVLpWkJlvj7+2PhwoVwc3NTZpnp0qULfvnlF2g0GmWWRQ970/mo2dvb48svv8SoUaOUWRY999xzD7SrT2kIDg7Gzp07LQbvLPnkk0/w9ddfF+saeprY2dlhypQpRX5+pk2bZvF6f1RUKhVGjhyJr776qsTv5+O+xtq1a4d///0Xbdq0UWYVSbl4rzVdb2q1GhMmTCiyPerXrx+++OIL2et/4YUX8Mknn8jKFVSjRg0sWrSoxCM4H5fAwECsWrUKFStWlNJK8/uFiIiIiP5bHkmwxwiVbGSPXmWLPGEHvcpOGqRzj1Cs12PMH9Vzr0T+OsxQCSNgyJ+vZRQQeiOMOj30hnvTJMTjmscFoH79+qhevbosrXPnzmY3TyWhUqnQt29fnDt3Dp999hnq1Kkj65g7OjqidevWWLFiBU6fPl3kQtB169bFoUOHMHXqVNmUMy8vL3zwwQf466+/ULVqVVmdwpiCW2fOnLH4uuzt7REQEIDRo0cjOjrabArL00itVmP69OmIiorC4MGDzQJjzs7OaNiwIebPn4+jR4+iTp06svzHqXr16jhw4ADWrFmD1q1bw9nZWcqzsbFBrVq1pLU2JkyYUKybuadRpUqVsHPnTgwZMgSOjo5A/u/XuHFjbN26FSNGjDALsD5qtra2eOedd6TPZK1atWSj3JydndG6dWusWbMGL7/8sqzu477GfH19sW7dOpw+fRoffPCB2ecUAMqXL4+GDRvis88+w+nTpy0Gh6zpeuvUqRP27t2Ldu3aSa/T3t4erVu3xtatW/Hdd99J15qJKTCydu1a1K1bV0qvUKECRo8ejZ07d8LPz09W53GoXr06wsPD0bVrV5QvX16WV758eXTt2hVr167FoUOHzL6bUMrfL0RERET036ESllYOfUD79vyL3Lt6bKm4GypNGajyb45sVAJaOx1sDGqEnn0JKhsb5Oh1EF8vQ+XbRtjdW3YZKhtbaN0qIk1tRN7HfZB0LRUJ11Ox6Big8vCByvbeEHub5+xhW8YBFYyZ+PxlZzg42KJi1adv3QgiInq0hg4ditWrV0v/VqvV2LZtG4KCgmTliIiIiIieJaU6sqdMmecgjEIa2aOTDjvkCHvkwQYGnQF6nR7CaMhfWfl/+2wJARj1BgiDAXfv5sKgN9wb3WM0QIgCW6/r9RB5Ohj0Buh1eVDODiMiIiIiIiIielaVarDHP7Ay7Gxs7y3QbLy33breqILOAOTobXDXIJCTlYeczDzACKiMRqhUkA4bFaDT5cKQl4e7OTrkZutg1Buh0hsBvQ4w6AGDHkKng7ibB73eCF2eEa7u5jvSEBERERERERE9i0o12OPs5oRqz1eB0SBgyMuF4W4ODHdzoL+bi9ycPGTf1eFWcg7SU3Kgu6MC9AYYhb7AkYe8uxkw5mYj84YOGTfuQpd5b1Fmld4Ild4Ald4A6PQQd/NgozfAy7c8HMs+p3wpRERERERERETPpFJds4eIiOhx4po9RERERETmSnVkDxERERERERERPVkM9hARERERERERWRFO4yIiIiIiIiIisiIc2UNEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPERERERER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zZqFqVOnwsfHB7t374anp6eyCBERET1jHsnIHiIiKp4mTZrgxRdfRFRUFCIjI5XZVIAQAkuXLkV6ejr69+8PX19fZRGiZ05qairWrl0LAKhZsyacnZ2VRYiIiOgZxGAPEdETVLZsWYwfPx4qlQqzZ8/GzZs3lUUoX1RUFFauXInAwEC88cYbxZ4iTGQNpk+fjlWrViEjIwPI38b3+PHj6NWrF2JjY6FSqTBw4ECUKVNGWZWIiIieQQz2EBE9YS1btsSnn36K6OhofPfdd4Wu9fEsy8jIwCeffIK7d+9i6tSpnNtOz5z4+Hi8++678PHxgVarhYuLC5o1a4bo6GgAwIgRI9ChQwdlNSIiInpGcc0eIiIioqfc7t278e233+LAgQPSFuqOjo5o3rw5Jk6ciODgYI52IyIiIgmDPUREREREREREVoTTuIiIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWRCWEEMrE+4mLi1MmERERERERERHRU+CBgj1ERERERERERPR04jQuIiIiIiIiIiIrwmAPEREREREREZEVYbCHiIiIiIiIiMiKMNhDRERERERERGRFGOwhIiIiIiIiIrIiDPYQEREREREREVkRBnuIiIiIiIiIiKwIgz1ERERERERERFaEwR4iIiIiIiIiIivCYA8RERERERERkRVhsIeIiIiIiIiIyIow2ENEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPEREREREREZEVYbCHiIiIiIiIiMiKPHCwJykpCTVr1oRWqzU7/P398dFHHyEtLU1ZDXPmzJGV/e2335RFZPR6PQYPHiyVr1mzJpKSkmRlDAYDtm3bhldffRVubm5SWU9PT/Tp0wdXr16Vyh49ehQeHh5mr9l0eHh4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IivCYA8RERERERERkRVhsIeIiIiIiIiIyIow2ENEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPEREREREREZEVYbCHiIiIiIiIiMiKqIQQQpl4P5mZmcokIiIiIiIiIiJ6CjxQsIeIiIiIiIiIiJ5OnMZFRERERERERGRFGOwhIiIiIiIiIrIiDPYQEREREREREVkRBnuIiIiIiIiIiKwIgz1ERERERERERFaEwR4iIiIiIiIiIivCYA8RERERERERkRVhsIeIiIiIiIiIyIow2ENEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPEREREREREZEVeazBHiEE1q1bh+rVq0Or1cLNzQ3/93//h5ycHGVRIiIiIiIiIiJ6AI812JOdnY2zZ89i69atuHnzJkaOHInZs2dj5cqVyqJERERERERERPQAVEIIoUx8XKKjo9GhQwd069YN4eHhymwiIiIiIiIiIiqhBx7Zk5SUhJo1a0Kr1Zod/v7++Oijj5CWlqasJnPx4kXk5OTAz89PmUVEVGJCCOzduxchISFSe9S+fXucOXNGWdSiEydOwNvb26xN02q1qFmzJpKSkqSyQgjs27cPbdu2hbOzM7RaLSpVqoRJkybhzp07svOa5OTkIDw8XJrK6uzsjLCwMFy7dk1ZlIieQmfOnEH79u2ldiEkJAR79+5FcZ6bpaSkoG7dumZti1arhYeHB44ePSqVHTp0qFkZ5VGwzoO0R0+S6fW2bt1a+n1q1aqF3377DQaDQVm8SDk5OVi9erWs3ff09MSnn36K3NxcWdk7d+5g0qRJqFSpErT5ywkMHjzY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Rffu3YW7u7vYv3+/lNahQwfh5OQkevfuLUaOHCkGDRok3N3dhUajEfPmzbN4M5GQkCCqV68uBg8ebDHfpKTt0ZNkNBrFnDlzhCb/gWDBw8nJSWzYsEEqW7Cvqez3paeni1dffdXsHJbOIwoE5pRlNRqN6Ny5syyIZ7oRU5YzHa+++qqsPBE9OaY+naurqxgwYIAYOXKkaNSokdBoNGLOnDlS22nqJ7q4uIiff/5Zqm8KHL/44oti5MiRYtiwYaJu3bpFftbv3Lkj2rZtK9q2bSvu3LmjzL6vpzHYI4QQGzZskALayqPgeykUAzsK/h55eXli9OjRZvULO49JYcEekd/ed+7c2excGo3GrI9Pco8s2COEED/88IP0h5g1a5bFP+yDYLCHiCzR6/Vi7dq1IjAwUGg0GuHq6iomTJhgNlrHUrAnLy9PhIeHi5o1a0rtloeHhxgxYoTFG6XY2FgRFhYmPDw8pPKNGjUSixcvFtnZ2criIiYmRup83O/cRPT0MRqNIjIyUvocOzk5iUGDBpl9hi0Fe4xGo9iwYYOsDXB1dRU9e/YUsbGxsvoF7d+/X7i4uIiZM2cqs2RK2h49aXq9Xqxbt040aNBAei979Ohh9l7k5eWJsWPHWhzZI/JH7CxatEhq8ws7j0lCQoIYMWKEcHV1FRqNRgQGBopFixaZvUdZWVli7ty50g2f6dwtW7YUv//+u9nrIKInKyEhQQwaNEgKVDRq1EhERkbK7j0LC/acO3dO9OzZU2oXTPXXrVsnjexTunTpkqhWrdp9A/GFeVqDPUajUfzzzz+iVatW0nvRqlUr8c8//5j9ntHR0RZH9ggLbXxR5zEpKtgjhBAZGRni008/lUZj+vj4iE8//dSsj09yKqGc4EhERERERERERP9Zpb5AMxERERERERERPTkM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPEREREREREZEVYbCHiIiIiIiIiMiKMNhDRERERERERGRFGOwhIiIiIiIiIrIiDPYQEREREREREVkRBnuIiIiIiIiIiKwIgz1ERERERERERFaEwR4iIiIiIiIiIivCYA8RERERERERkRVhsIeIiIiIiIiIyIow2ENEREREREREZEUY7CEiIiIiIiIisiIM9hARERERERERWREGe4iIiIiIiIiIrAiDPUREREREREREVoTBHiIiIiIiIiIiK8JgDxERERERERGRFWGwh4iIiIiIiIjIijDYQ0RERERERERkRRjsISIiIiIiIiKyIgz2EBERERERERFZEQZ7iIiIiIiIiIisCIM9RERERERERERWhMEeIiIiIiIiIiIrwmAPEREREREREZEVYbCHiIiIiIiIiMiK/D8J4BVc0pn2igAAAABJRU5ErkJggg==)"""

