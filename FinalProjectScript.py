import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


st.title('Final Project')
st.write('### Nick Reese')

s = pd.read_csv('social_media_usage.csv')
print(s)

### Cleaning the data:
def clean_sm(x):
    return np.where(x== 1, 1, 0)


ss = s.copy()
ss['sm_li'] = clean_sm(ss['web1h'])
ss = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age','sm_li']]
ss.rename(columns = {'educ2': 'education'}, inplace = True)
# Dropping values 
ss = ss[ss['income'] <= 9]
ss = ss[ss['education'] <= 8]
ss = ss[ss['age'] <= 98]
ss.dropna(inplace = True)

ss['female'] = np.where(ss['gender'] == 2, 0, 1)
ss['parent'] = np.where(ss['par'] == 2, 0, 1)
ss['married'] = np.where(ss['marital'] >= 2, 0, 1)
ss = ss.drop(['par', 'marital', 'gender'], axis =1)


### Eduction - Streamlit 1
education = st.selectbox('Education level',
                         options = ['Less than high school',
                                    'High school incomplete with NO diploma',
                                    'High school graduate Grade 12 with diploma or GED certificate',
                                    'Some college, no degree',
                                    'Two-year associate degree from a college or university',
                                    'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)',
                                    'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                                    'Postgraduate or professional degree, including master’s, doctorate, medical or law'])
st.write(f'Euducation: {education}')

if education == 'Less than high school':
    education = 1
elif education == 'High school incomplete with NO diploma':
    education = 2
elif education == 'High school graduate Grade 12 with diploma or GED certificate':
    education = 3
elif education == 'Some college, no degree':
    education = 4
elif education == 'Two-year associate degree from a college or university':
    education = 5
elif education == 'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)':
    education = 6
elif education == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    education = 7
else:
    education = 8
st.write(f'Education: {education}')

### Age - Streamlit 2

age = st.slider(label= "Enter Your Age",
                min_value= 1,
                max_value= 98,
                value= 30)
st.write('Your Age:', age)

### Side Bar for income, female, parent and married 
with st.sidebar:
    income = st.number_input('Income (low = 1 to high = 9)', 1,9)
    female = st.number_input("Female (0=no, 1=yes)", 0, 1)
    parent = st.number_input("Parent (0=no, 1 = yes)", 0, 1)
    married = st.number_input("Married (0=no, 1=yes)", 0,1)


#Income
if income <= 3:
    income_label = "Low Income"
elif income > 3 and income < 7:
    income_label = "Middle Income"
else:
    income_label = "High Income"

#Gender
if female == 1:
    female_label = "Female"
else:
    female_label = "Male"

#Parent 
if parent == 1:
    parent_label = "Children"
else:
    parent_label = " no Children"

#Married
if married == 1:
    married_label = "Married"
else:
    married_label = "Single"

st.write(f"You are {married_label}, that makes {income_label}, who is a {female_label}, and have {parent_label}") 



st.write('### Now we are going to look at the Social Media Data')
#Data 
st.write(ss.head(10))

st.write('If you would like to see some of the max values of the dataset, check the box!')
if st.checkbox("Max Values"):
    st.dataframe(ss.style.highlight_max(axis=0, color='green'))

#st.write('### Correlation  of our Data')
#st.write("If you would like to see the correlations, please check the box")

corr_ = ss.corr()
#sns.heatmap(corr_, annot=True, cmap='coolwarm', fmt=".2f")
#st.pyplot()

# Check Box for Correlation 
#import seaborn as sns
#if st.checkbox("Correlation  for Variables"):
#  sns.heatmap(corr_, annot=True, cmap='coolwarm', fmt=".2f")
#  st.pyplot()

st.write('Make sure you put in all your values to see if the prediciton was correct!')

### Creating Logistic Regression:
y = ss['sm_li']
x = ss[['age', 'income', 'education', 'female', 'parent', 'married']]

ss = ss.dropna()
x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 987)





lr = LogisticRegression(class_weight = 'balanced')
# Fitting the Algorith to training data:
lr.fit(x_train, y_train)

person = [age, income, education, female, parent, married]
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

st.write('After you put in all your stats, lets see if we guessed correctly whether or not you have a LinkedIn!')

if predicted_class == 1:
    predicted = (f" You have a {(probs[0][1]*100).round(2)}% chance of being a LinkedIn User")
else:
    predicted = (f"There is a {(probs[0][1]*100).round(2)}% chance of not being a LinkedIn User")

if st.button('Find Out Here'):
    st.write(predicted)

st.write('Please feel free to grade my assignment with the in the select box below')


Final_Grade = st.selectbox('Final Grade',
                           options= ['A', 'B', 'C', 'D'])





if Final_Grade == 'A':
    Final_Grade_label = "That is correct!"
elif Final_Grade == 'B':
    Final_Grade_label = 'Close but try again'
elif Final_Grade == 'C':
    Final_Grade_label = 'Yeah you are going the wrong direction'
elif Final_Grade == 'D':
    Final_Grade_label = "You are breaking my heart please try again"

st.write(Final_Grade_label)
