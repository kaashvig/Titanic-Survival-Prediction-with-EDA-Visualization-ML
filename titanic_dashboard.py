import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Load and clean dataset
# --------------------------
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

# --------------------------
# Streamlit UI
# --------------------------
st.title("Titanic Survival Prediction ")

# User input
pclass = st.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("Parents/Children aboard", 0, 6, 0)
fare = st.slider("Fare", 0.0, 512.0, 32.0)
embarked = st.selectbox("Embarked", ["C","Q","S"])

# Encode input
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

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")
