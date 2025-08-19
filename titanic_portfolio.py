# titanic_portfolio.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------
# Step 1: Load Data
# --------------------------
df = pd.read_csv('train.csv')
print("First 5 rows of the dataset:\n", df.head())

# --------------------------
# Step 2: Data Cleaning
# --------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

print("\nData after cleaning:\n", df.head())
print("\nData info:\n")
df.info()

# --------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')
plt.show()

sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Fare Distribution')
plt.show()

# --------------------------
# Step 4: Feature Encoding for ML
# --------------------------
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Features and target
X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Step 5: Build Random Forest Model
# --------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------
# Step 6: Evaluate Model
# --------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
