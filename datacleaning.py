import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Load Data
# --------------------------
df = pd.read_csv('train.csv')

# --------------------------
# Step 2: Data Cleaning
# --------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

# --------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------

# Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival by Passenger Class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Fare Distribution')
plt.show()
