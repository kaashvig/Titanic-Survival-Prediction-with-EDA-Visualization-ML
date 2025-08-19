# Titanic Survival Prediction Project


## **Project Overview**
This is an **end-to-end Data Science project** aimed at predicting the survival of passengers aboard the Titanic using Python. The project includes:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA) & Visualizations  
- Machine Learning Model (Random Forest)  
- Interactive Streamlit Dashboard  

---

## **Dataset**
The dataset is from Kaggle: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

It contains passenger information such as age, sex, class, fare, and survival status.

---

## **Features Used**
- PassengerId  
- Pclass  
- Name  
- Sex  
- Age  
- SibSp  
- Parch  
- Ticket  
- Fare  
- Embarked  

---

## **Methodology**
1. **Data Cleaning**
   - Filled missing `Age` values with median.  
   - Filled missing `Embarked` values with mode.  
   - Dropped unnecessary columns for modeling.  

2. **Exploratory Data Analysis**
   - Visualized survival count based on sex, class, and age.  
   - Studied correlations between features.  

3. **Machine Learning Model**
   - Used **Random Forest Classifier** for prediction.  
   - Trained on the cleaned dataset and evaluated accuracy.  

4. **Streamlit Dashboard**
   - Created an interactive dashboard to input passenger data and get survival prediction.  

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/kaashvig/Titanic-Survival-Prediction-with-EDA-Visualization-ML.git
cd Titanic-Survival-Prediction-with-EDA-Visualization-ML
