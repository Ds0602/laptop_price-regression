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

mae_values = {}
mse_values = {}

for i in range(5,16):
    model = DecisionTreeRegressor(random_state=1, max_depth=i)
    model.fit(X_encoded, y_train)
    #training the model for different max_depth values

    #testing the model
    target_values = list(y_test)
    predicted_values = model.predict(X_test_encoded)

    mae = mean_absolute_error(target_values, predicted_values)
    mae_values[i] = mae
    mse = mean_squared_error(target_values, predicted_values)
    mse_values[i] = mse
    print("Max Depth: ", i)
    print("MAE: ", mae)
    print("MSE: ", mse,"\n")

print("Best MAE: ", min(mae_values, key=mae_values.get), "with value: ", min(mae_values.values()))
print("Best MSE: ", min(mse_values, key=mse_values.get), "with value: ", min(mse_values.values()), "square rooted:" ,min(mse_values.values())**0.5)
