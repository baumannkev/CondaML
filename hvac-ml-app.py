import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
future_time = 1
size_window = 1
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='BCIT HVAC Machine Learning App',
    layout='wide')

st.write("""
# BCIT Room Temperature Prediction App

This app predicts the **room temperatures** at BCIT!

Data obtained from Kaizen CopperTreeAnalytics
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # island = st.sidebar.selectbox('Building',('NE01','SE12','SW3'))
        # sex = st.sidebar.selectbox('Room',('412','301'))
        # bill_length_mm = st.sidebar.slider('Parameter 1', 32.1,59.6,43.9)
        # bill_depth_mm = st.sidebar.slider('Parameter 2', 13.1,21.5,17.2)
        # flipper_length_mm = st.sidebar.slider('Parameter 3', 172.0,231.0,201.0)
        # body_mass_g = st.sidebar.slider('Parameter 4', 2700.0,6300.0,4207.0)
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Inputs features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
# BCIT Room Temperature Prediction App

This app predicts the **room temperatures** at BCIT!

Data obtained from Kaizen CopperTreeAnalytics
""")

df=pd.read_csv("Trend_Logs.csv", parse_dates=True )
df.columns

df = df.drop(['NE01_AHU7_EFS_POLL_TL','NE01_AHU7_SAT_POLL_TL', 'NE01_AHU7_EF_VFD_AL_COV_TL','NE01_AHU7_BSP_POLL_TL', 'NE01_AHU7_WEST_DSP_POLL_TL',
'NE01_AHU7_EAST_DSP_POLL_TL'],axis=1)

df['Timestamp']=pd.to_datetime(df['Timestamp'])

df = df.resample('15min', on='Timestamp').mean()

st.write(df['NE01_AHU7_RESET_POLL_TL'].sum())

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f')