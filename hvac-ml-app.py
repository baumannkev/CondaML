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


# Testing Sirine's Model 

# def plot_predictions(test, predicted, title):
#     plt.figure(figsize=(32,8))
#     plt.plot(test, color='blue',label='Actual Supply Air Temperature')
#     plt.plot(predicted, alpha=0.7, color='red',label='Predicted Supply Air Temperature')
#     plt.title(title)
#     plt.xlabel('Time')
#     plt.ylabel('Temperature')
#     plt.legend()
#     plt.show()

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

df= df.drop(['NE01_AHU7_EF_SPD_POLL_TL'],axis=1)
df = df.drop(['NE01_AHU7_RAT_POLL_TL'],axis=1)

# df.to_csv(r'Pre_processed2016_2019.csv')
df['Weekend']=(df.index.dayofweek // 5 == 1).astype(float)
df['Day']=((6<=df.index.hour) & (df.index.hour <=18))
df['Before_night']=((19<=df.index.hour) & (df.index.hour <=22))
df['night']=((df.index.hour>=23) & (df.index.hour <=5))
df['month'] = df.index.month

# df.to_csv(r'Pre_processed2016_2019_new.csv')
df = pd.get_dummies(df, columns=['month'])
df.describe()
temp = df["NE01_AHU7_HC_SAT_POLL_TL"].where((df.Day == 1) & (df.Weekend == 0.0) & (df.month_2 == 1)).count()

# Timer starts
starttime = time.time()
lasttime = starttime
lapnum = 1
value = ""
st.write("""
    ## Training..
""")
#use all data previous to 2019 for training and validation
df_train = df[(df.index.year < 2019)]
df_train.shape

#reserve the last year for testing
df_test = df[(df.index.year >= 2019)]
df_test.shape

import copy
import numpy as np
X = []
y = []
tmp_ret = []
i = 0
while i < len(df_train):
    row = df_train.iloc[i]
    if len(tmp_ret) == size_window:
        if i + (future_time - 1) < len(df_train):
            X.append(np.array(copy.deepcopy(tmp_ret)))
            y.append(df_train.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
    tmp_ret.append(copy.deepcopy(row.to_list()))
    if len(tmp_ret) == size_window + 1:
        tmp_ret.pop(0)
    i += 1

X = np.array(X)
y = np.array(y)
X.shape

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,shuffle=True, random_state=42)
X_test = []
y_test = []
tmp_ret = []
i = 0
while i < len(df_test):
    row = df_test.iloc[i]
    if len(tmp_ret) == size_window:
        if i + (future_time - 1) < len(df_test):
            X_test.append(np.array(copy.deepcopy(tmp_ret)))
            y_test.append(df_test.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
    tmp_ret.append(copy.deepcopy(row.to_list()))
    if len(tmp_ret) == size_window + 1:
        tmp_ret.pop(0)
    i += 1

X_test = np.array(X_test)
y_test = np.array(y_test)

y_test.shape

X_train.shape

X_val.shape
st.write("""
    ## Done Training
""")
# Total time elapsed since the timer started
totaltime = round((time.time() - starttime), 2)
st.write("Time taken = " + str(totaltime) + " seconds")

st.write("""
    ## Model 1: Extra Trees for Regression
""")

#################################### Model 1: Extra Trees for Regression #################################### 
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor
# define the model
model_Extra = ExtraTreesRegressor()
#fit the model on train data
model_Extra.fit(X_train.reshape(X_train.shape[0],-1), y_train)

pred_tree_val = model_Extra.predict(X_val.reshape(X_val.shape[0],-1))
st.write(pred_tree_val)
st.write ('Mean Squared Error on Val Set = ', mean_squared_error(y_val,pred_tree_val))
st.write ('Mean Absolute Error on Val Set = ', mean_absolute_error(y_val,pred_tree_val))
pred_tree_test = model_Extra.predict(X_test.reshape(X_test.shape[0],-1))
st.write(pred_tree_test)
st.write ('Mean Squared Error on Test Set = ', mean_squared_error(y_test,pred_tree_test))
st.write ('Mean Absolute Error on Test Set = ', mean_absolute_error(y_test,pred_tree_test))

# plot_predictions(y_test, pred_tree_test, "Predictions made by ExtraTree model")

