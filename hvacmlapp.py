import streamlit as st
import pandas as pd
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
from sklearn.model_selection import train_test_split
import time
# import matplotlib as plt
# import seaborn as sns
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout='wide')
def app():
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


    

