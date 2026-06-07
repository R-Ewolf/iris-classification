from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

print(df.head())

from sklearn.model_selection import train_test_split

X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

import numpy as np

example = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(example)
print("Предсказание класс:", iris.target_names[prediction[0]])
