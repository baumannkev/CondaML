import streamlit as st
import pandas as pd
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
future_time = 1
size_window = 1
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor
import time

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    # X = df.iloc[:,:-1] # Using all column except for the last column as X
    # Y = df.iloc[:,-1] # Selecting the last column as Y

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

    df['Weekend']=(df.index.dayofweek // 5 == 1).astype(float)
    df['Day']=((6<=df.index.hour) & (df.index.hour <=18))
    df['Before_night']=((19<=df.index.hour) & (df.index.hour <=22))
    df['night']=((df.index.hour>=23) & (df.index.hour <=5))
    df['month'] = df.index.month
    df = pd.get_dummies(df, columns=['month'])
    temp = df["NE01_AHU7_HC_SAT_POLL_TL"].where((df.Day == 1) & (df.Weekend == 0.0) & (df.month_2 == 1)).count()
    
    #use all data previous to 2019 for training and validation
    df_train = df[(df.index.year < 2019)]
    df_train.shape  

    #reserve the last year for testing
    df_test = df[(df.index.year >= 2019)]
    df_test.shape

    X = []
    Y = []
    tmp_ret = []
    i = 0
    while i < len(df_train):
        row = df_train.iloc[i]
        if len(tmp_ret) == size_window:
            if i + (future_time - 1) < len(df_train):
                X.append(np.array(copy.deepcopy(tmp_ret)))
                Y.append(df_train.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
        tmp_ret.append(copy.deepcopy(row.to_list()))
        if len(tmp_ret) == size_window + 1:
            tmp_ret.pop(0)
        i += 1

    X = np.array(X)
    Y = np.array(Y)

    # Data splitting ------------------------------------------------------------------------------
    X_train, X_val, Y_train, y_val = train_test_split(X,Y,test_size=0.2,shuffle=True, random_state=42)
    
    X_test = []
    Y_test = []
    tmp_ret = []
    i = 0
    while i < len(df_test):
        row = df_test.iloc[i]
        if len(tmp_ret) == size_window:
            if i + (future_time - 1) < len(df_test):
                X_test.append(np.array(copy.deepcopy(tmp_ret)))
                Y_test.append(df_test.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
        tmp_ret.append(copy.deepcopy(row.to_list()))
        if len(tmp_ret) == size_window + 1:
            tmp_ret.pop(0)
        i += 1

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(X)
    st.write('Y variable')
    st.info(Y)

    etr = ExtraTreesRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_depth=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    etr.fit(X_train.reshape(X_train.shape[0],-1), Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = etr.predict(X_train)
    # st.write('Coefficient of determination ($R^2$):')
    # st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = etr.predict(X_test)
    # st.write('Coefficient of determination ($R^2$):')
    # st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(etr.get_params())

#---------------------------------#
st.write("""
# BCIT HVAC Machine Learning Model Builder
In this implementation, the *ExtraTreesRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)