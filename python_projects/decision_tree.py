import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/laptop_data (1).csv")
#uploading the data


features = ["Company", "TypeName", "Inches", "ScreenResolution", "Ram", "Memory", "Weight", "Cpu", "Gpu", "OpSys"]
X = data[features]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


X_encoded = pd.get_dummies(X_train, columns=features)
X_test_encoded = pd.get_dummies(X_test, columns=features).reindex(columns=X_encoded.columns, fill_value=0)
#enconding the features

model = DecisionTreeRegressor(random_state=1, max_depth=13)
model.fit(X_encoded, y_train)

y_pred = model.predict(X_test_encoded)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse, "Square rooted MSE:", mse**0.5)
print("R-squared:", model.score(X_test_encoded, y_test))