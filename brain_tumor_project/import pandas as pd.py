from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample data
data = {
    'amount':[100,5000,200,7000,300],
    'time':[1,2,3,4,5],
    'fraud':[0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df[['amount','time']]
y = df['fraud']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train,y_train)

prediction = model.predict([[6000,2]])
print("Fraud Prediction:", prediction)