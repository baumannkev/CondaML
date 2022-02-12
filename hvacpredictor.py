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
def app():
    st.write("""
    # BCIT Room Temperature Prediction App

    This app predicts the **room temperatures** at BCIT!

    Data obtained from Kaizen CopperTreeAnalytics
    """)

    st.sidebar.header('User Input Features')

    room = st.sidebar.selectbox('Room',('Room 412','Room 411','Room 410', 'Room 409', 'Room 408', 'Room 407', 'Room 415D'))
          
    st.sidebar.header('3. Prediction Example')
    cool_coil_valve = st.sidebar.slider(
        'Cooling Coil Valve (% open)', 0.0, 100.0, 37.6130981)  # 1
    hot_water_valve = st.sidebar.slider(
        'Hot Water Valve (% open)', 0.0, 100.0, 0.0)  # 2
    hot_water_supply_temp = st.sidebar.slider(
        'Hot Water Supply Temperature (°C)', 0.0, 100.0, 19.9593983)  # 3
    hot_water_return_temp = st.sidebar.slider(
        'Hot Water Return Temperature (°C)', 0.0, 100.0, 20.8191605)  # 4
    dampers_pos = st.sidebar.slider(
        'Dampers Position (% open)', 0.0, 100.0, 20.4688644)  # 5
    supply_air_temp = st.sidebar.slider(
        'Supply Air Temperature (°C)', 0.0, 100.0, 19.946228)  # 6
    mixed_air_temp = st.sidebar.slider(
        'Mixed Air Temperature (°C)', 0.0, 100.0, 20.8940868)  # 7
    return_air_temp = st.sidebar.slider(
        'Return Air Temperature (°C)', 0.0, 100.0, 21.5931683)  # 8
    supply_fan_speed = st.sidebar.slider(
        'Supply Fan Speed (% of max speed)', 0.0, 100.0, 65.0570679)  # 9
    return_fan_speed = st.sidebar.slider(
        'Return Fan Speed (% of max speed)', 0.0, 100.0, 53.0570641)  # 10
    outside_air_temp = st.sidebar.slider(
        'Outside Air Temperature (°C)', -20.0, 45.0, 19.1146049)  # 11



    if st.button('Press to Predict'):
                st.write('Try adjusting the hyperparameters')
    

