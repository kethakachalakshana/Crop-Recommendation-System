import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# just checking if there are any missing values before moving forward
# print(df.isnull().sum())

# plot rainfall requirements for different crops
plt.figure(figsize=(12, 5))
sns.barplot(x='label', y='rainfall', data=df, palette='viridis')
plt.xticks(rotation=45) 
plt.title('Rainfall Requirement by Crop')
plt.xlabel('Crop Type')
plt.ylabel('Rainfall (mm)')
plt.show()

# drop the target column to set up features
X = df.drop('label', axis=1) 
y = df['label']

# splitting into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# building the random forest classifier (using 100 trees)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# testing the model accuracy
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Model Accuracy is: {round(acc * 100, 2)}%")

# custom testing with some sample dummy data
sample_data = pd.DataFrame({
    'N': [90], 'P': [42], 'K': [43], 
    'temperature': [20.8], 'humidity': [82.0], 
    'ph': [6.5], 'rainfall': [202.9]
})

pred = model.predict(sample_data)
print(f"Recommended Crop: {pred[0].upper()}")
