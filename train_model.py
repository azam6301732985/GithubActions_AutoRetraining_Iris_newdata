import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data/iris_old.csv')
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

acc = clf.score(X_test, y_test)
print(f"Training Accuracy on old data: {acc:.2f}")
