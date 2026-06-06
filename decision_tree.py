import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("data/laptop_data (1).csv")
#uploading the data

y = data["Price"].head(1000)
y_test = data["Price"].tail(304)
#separating the target into y and y_test for training and testing respectively

features = ["Company", "TypeName", "Inches", "ScreenResolution", "Ram", "Memory", "Weight", "Cpu", "Gpu", "OpSys"]
#selecting features
X = data[features].head(1000)
X_test = data[features].tail(304)
#separating the features into X and test for training and testing respectively


X_encoded = pd.get_dummies(X, columns=features)
X_test_encoded = pd.get_dummies(X_test, columns=features)
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)
#enconding the features

model = DecisionTreeRegressor(random_state=1)
model.fit(X_encoded, y)
#training the model

#testing the model
print(model.predict(X_test_encoded.head(5)))
print(y_test.head(5))