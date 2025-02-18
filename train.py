import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Crop_recommendation.csv")

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

# Save the model to a file

pickle.dump(model, open('model.pkl', 'wb'))

# accuracy = model.score(x_test, y_test)*100
# print(f"This Model Accuracy is: {accuracy:.2f}%")
A