import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# separate features and target
X = df.drop('label', axis=1)
y = df['label']

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# init and train the random forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# evaluate accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model training completed.")
print(f"Accuracy Score: {acc * 100:.2f}%\n")

# testing the model with some random sample data
# feel free to change these values and run again
sample_data = pd.DataFrame({
    'N': [90], 'P': [42], 'K': [43], 
    'temperature': [20.8], 'humidity': [82.0], 
    'ph': [6.5], 'rainfall': [202.9]
})

pred = clf.predict(sample_data)
print(f"Recommended crop: {pred[0].upper()}")
