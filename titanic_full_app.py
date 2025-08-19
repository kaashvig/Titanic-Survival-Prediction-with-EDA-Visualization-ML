
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival App", layout="wide")


# Load and Clean Data

df = pd.read_csv('train.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Features and target
X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


# Sidebar for User Input

st.sidebar.header("Passenger Info Input")
pclass = st.sidebar.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1,2,3])
sex = st.sidebar.selectbox("Sex", ["male","female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 512.0, 32.0)
embarked = st.sidebar.selectbox("Embarked", ["C","Q","S"])

input_df = pd.DataFrame({
    'PassengerId':[0],
    'Pclass':[pclass],
    'Age':[age],
    'SibSp':[sibsp],
    'Parch':[parch],
    'Fare':[fare],
    'Sex_male':[1 if sex=='male' else 0],
    'Embarked_Q':[1 if embarked=='Q' else 0],
    'Embarked_S':[1 if embarked=='S' else 0]
})

# Main Page

st.title("Titanic Survival Prediction & EDA")

# EDA Visualizations
st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("**Survival by Gender**")
    fig1 = sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig1.figure)
    fig1.clear()

with col2:
    st.write("**Survival by Passenger Class**")
    fig2 = sns.countplot(x='Survived', hue='Pclass', data=df)
    st.pyplot(fig2.figure)
    fig2.clear()

st.write("**Age Distribution**")
fig3 = sns.histplot(df['Age'], bins=20, kde=True)
st.pyplot(fig3.figure)
fig3.clear()

st.write("**Fare Distribution**")
fig4 = sns.histplot(df['Fare'], bins=20, kde=True)
st.pyplot(fig4.figure)
fig4.clear()

#Survival Prediction
st.subheader("Predict Survival for a Passenger")

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")
