import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Dell\Documents\extraction\vlsi.csv")
df = df.drop(['Design'], axis=1)  


x = df.drop(['Power(mW)'], axis=1)
y = df['Power(mW)']


sc_x = StandardScaler()
sc_y = StandardScaler()

x_scaled = sc_x.fit_transform(x)
x_scaled = pd.DataFrame(data=x_scaled, columns=x.columns)

y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1))
y_scaled = pd.DataFrame(data=y_scaled)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)



lr = XGBRegressor()
lr.fit(x_train, y_train.values.ravel())  


#1
y_pred = lr.predict(x_test)
#2
y_train_predicted=lr.predict(x_train)
#3
#y_pred_real = sc_y.inverse_transform(y_pred.reshape(-1,1))
#4
#y_test_real = sc_y.inverse_transform(y_test.values.reshape(-1,1))

print("r2_score_test:", r2_score(y_test, y_pred))
print("mean squared error:", mean_squared_error(y_test, y_pred))
print("mean absolute error:", mean_absolute_error(y_test, y_pred))

r2_train = r2_score(y_train, y_train_predicted)
print("r2-train=",r2_train)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted Plot")
plt.legend()
plt.grid(True)
plt.show()


def prediction(LUTs, FFs, DSPs, IOs, Bit_Width, Pipeline, Unroll, Delay):
    features = np.array([[LUTs, FFs, DSPs, IOs, Bit_Width, Pipeline, Unroll, Delay]], dtype=float)
    features_scaled = sc_x.transform(features)
    prediction_scaled = lr.predict(features_scaled)
    prediction_real = sc_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    return prediction_real[0][0]


predy = prediction(410,200,2,10,8,0,1,5.10)
print(predy)


pickle.dump(lr, open("vlsi.pkl", 'wb'))
with open("sc_x.pkl", "wb") as f:
    pickle.dump(sc_x, f)
with open("sc_y.pkl", "wb") as f:
    pickle.dump(sc_y, f)
