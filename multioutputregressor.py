import streamlit as st
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

import json
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from math import sqrt

def app():
    #---------------------------------#
    # Page layout
    ## Page expands to full width
    #---------------------------------#
    def plot_predictions(test, predicted, title):
        fig = plt.figure(figsize=(32, 8))
        plt.title(title)
        plt.plot(test, color='blue',label='Actual Supply Air Temperature')
        plt.plot(predicted, color='red',label='Predicted Supply Air Temperature')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        st.balloons()
        st.pyplot(fig)

    # Model building
    def build_model(df):

        # Reserve 20% of the data for testing
        
        with st.spinner(text='Training model'):
            train_rows, test_rows = train_test_split(df['rows'], test_size=0.2, shuffle=True, random_state=2022_02_02)

            st.markdown('**1.2. Data splits**')
            st.write('Training set')
            st.success(f'Number of training data: {len(train_rows)}')
            st.write('Test set')
            st.success(f'Number of testing data: {len(test_rows)}')

            # Both are tuples of lists of values
            train_input_points, train_output_points = list(zip(*train_rows))

            #Create Model
        
        with st.spinner(text='Creating model'):
            pipeline = make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(SGDRegressor(max_iter=1_000_000, tol=1e-6))
            )

            #Fit model to training datasets
            pipeline.fit(train_input_points, train_output_points)

        #Test model_selection
        st.subheader('2. Model Testing')
        with st.spinner(text='Testing model_selection'):
            all_mean_absolute_errors = []
            all_mean_squared_errors = []

            for [inputs, outputs] in test_rows:
                [prediction] = pipeline.predict([inputs])

                absolute_errors = [abs(p - o) for p,o in zip(prediction, outputs)]
                squared_errors = [(p-o)*(p-o) for p,o in zip(prediction, outputs)]

                mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
                mean_squared_error = sum(squared_errors) / len(squared_errors)

                all_mean_absolute_errors.append(mean_absolute_error)
                all_mean_squared_errors.append(mean_squared_error)

        st.markdown('**2.1. Training set**')
        mean_mean_absolute_error = sum(all_mean_absolute_errors) / len(all_mean_absolute_errors)
        root_mean_mean_squared_error = sqrt(sum(all_mean_squared_errors) / len(all_mean_squared_errors))

        st.write('Error (MSE or MAE):')

        st.success(f'Mean absolute error: {mean_mean_absolute_error} °C')
        st.success(f'Root mean squared error: {root_mean_mean_squared_error}')

        #Prediction Example
        st.markdown('**2.2. Prediction Example**')
        [prediction] = pipeline.predict([[
            37.6130981,
            0.0,
            19.9593983,
            20.8191605,
            20.4688644,
            19.946228,
            20.8940868,
            21.5931683,
            65.0570679,
            53.0570641,
            19.1146049
        ]])

        for i in range(0, len(prediction)):
            column_name = df['columns'][1][i][1]
            st.info(f'{column_name} -> {prediction[i]} °C')

        # --------------------------------- #
        # df = df.drop(['NE01_AHU7_EFS_POLL_TL','NE01_AHU7_SAT_POLL_TL', 'NE01_AHU7_EF_VFD_AL_COV_TL','NE01_AHU7_BSP_POLL_TL', 'NE01_AHU7_WEST_DSP_POLL_TL',
        # 'NE01_AHU7_EAST_DSP_POLL_TL'],axis=1)
        # st.write(df.head(5))
        # df['Timestamp']=pd.to_datetime(df['Timestamp'])
        
        # df = df.resample('15min', on='Timestamp').mean()
        
        # # st.write(df['NE01_AHU7_RESET_POLL_TL'].sum())

        # df.isnull().sum()

        # df = df.dropna()

        # df.isnull().sum()

        # corrmat = df.corr()

        # df= df.drop(['NE01_AHU7_EF_SPD_POLL_TL'],axis=1)
        # df = df.drop(['NE01_AHU7_RAT_POLL_TL'],axis=1)

        # df['Weekend']=(df.index.dayofweek // 5 == 1).astype(float)
        # df['Day']=((6<=df.index.hour) & (df.index.hour <=18))
        # df['Before_night']=((19<=df.index.hour) & (df.index.hour <=22))
        # df['night']=((df.index.hour>=23) & (df.index.hour <=5))
        # df['month'] = df.index.month
        # df = pd.get_dummies(df, columns=['month'])
        # temp = df["NE01_AHU7_HC_SAT_POLL_TL"].where((df.Day == 1) & (df.Weekend == 0.0) & (df.month_2 == 1)).count()
        # # st.write(df.head(5))
        # #use all data previous to 2019 for training and validation
        # df_train = df[(df.index.year < 2019)]
        # # df_train.shape  

        # #reserve the last year for testing
        # df_test = df[(df.index.year >= 2019)]
        # # df_test.shape

        # X = []
        # Y = []
        # tmp_ret = []
        # i = 0
        # while i < len(df_train):
        #     row = df_train.iloc[i]
        #     if len(tmp_ret) == size_window:
        #         if i + (future_time - 1) < len(df_train):
        #             X.append(np.array(copy.deepcopy(tmp_ret)))
        #             Y.append(df_train.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
        #     tmp_ret.append(copy.deepcopy(row.to_list()))
        #     if len(tmp_ret) == size_window + 1:
        #         tmp_ret.pop(0)
        #     i += 1

        # X = np.array(X)
        # Y = np.array(Y)

        # # Data splitting ------------------------------------------------------------------------------
        # X_train, X_val, Y_train, y_val = train_test_split(X,Y,test_size=0.2,shuffle=True, random_state=42)
        
        # X_test = []
        # Y_test = []
        # tmp_ret = []
        # i = 0
        # while i < len(df_test):
        #     row = df_test.iloc[i]
        #     if len(tmp_ret) == size_window:
        #         if i + (future_time - 1) < len(df_test):
        #             X_test.append(np.array(copy.deepcopy(tmp_ret)))
        #             Y_test.append(df_test.iloc[i+(future_time - 1)]['NE01_AHU7_HC_SAT_POLL_TL'])
        #     tmp_ret.append(copy.deepcopy(row.to_list()))
        #     if len(tmp_ret) == size_window + 1:
        #         tmp_ret.pop(0)
        #     i += 1

        # X_test = np.array(X_test)
        # Y_test = np.array(Y_test)

        # st.markdown('**1.2. Data splits**')
        # st.write('Training set')
        # st.info(X_train.shape)
        # st.write('Test set')
        # st.info(X_test.shape)

        # st.markdown('**1.3. Variable details**:')
        # st.write('X variable')
        # st.info(X)
        # st.write('Y variable')
        # st.info(Y)

        # etr = ExtraTreesRegressor(n_estimators=parameter_n_estimators,
        #     random_state=parameter_random_state,
        #     max_features=parameter_max_features,
        #     criterion=parameter_criterion,
        #     min_samples_split=parameter_min_samples_split,
        #     min_samples_leaf=parameter_min_samples_leaf,
        #     bootstrap=parameter_bootstrap,
        #     oob_score=parameter_oob_score,
        #     n_jobs=parameter_n_jobs)
        # etr.fit(X_train.reshape(X_train.shape[0],-1), Y_train)

        # st.subheader('2. Model Performance')

        # st.markdown('**2.1. Training set**')
        
        # pred_tree_val = etr.predict(X_val.reshape(X_val.shape[0],-1))
        # # Y_pred_train = etr.predict(X_train)
        
        # # st.write('Coefficient of determination ($R^2$):')
        # # st.info( r2_score(Y_train, Y_pred_train) )

        # st.write('Error (MSE or MAE):')
        # st.write('MSE:')
        # st.info(mean_squared_error(y_val,pred_tree_val))
        # st.write('MAE:')
        # st.info(mean_absolute_error(y_val,pred_tree_val))

        # st.markdown('**2.2. Test set**')
        # pred_tree_test = etr.predict(X_test.reshape(X_test.shape[0],-1))

        # # Y_pred_test = etr.predict(X_test)
        # # st.write('Coefficient of determination ($R^2$):')
        # # st.info( r2_score(Y_test, Y_pred_test) )

        # st.write('Error (MSE or MAE):')
        # st.write('MSE:')
        # st.info(mean_squared_error(Y_test,pred_tree_test))
        # st.write('MAE:')
        # st.info(mean_absolute_error(Y_test,pred_tree_test))

        # # st.info( mean_squared_error(Y_test, Y_pred_test) )

        # st.subheader('3. Model Parameters')
        # st.write(etr.get_params())

        # # def plot_predictions(test, predicted, title):
        # # plt.figure(figsize=(32,8))
        # # plt.plot(test, color='blue',label='Actual Supply Air Temperature')
        # # plt.plot(predicted, alpha=0.7, color='red',label='Predicted Supply Air Temperature')
        # # plt.title(title)
        # # plt.xlabel('Time')
        # # plt.ylabel('Temperature')
        # # plt.legend()
        # # plt.show()
        # plot_predictions(Y_test, pred_tree_test, "Predictions made by ExtraTree model")

    #---------------------------------#
    st.write("""
    # BCIT Room Temperature Machine Learning Model Builder App using Multioutput Regressor
    In this implementation, the *Multioutput Regressor()* function is used in this app for build a regression model using the **SGDRegressor** algorithm.
    Try adjusting the hyperparameters!
    """)

    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your JSON data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input JSON file", type=["JSON"])
        
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
        parameter_criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.sidebar.selectbox('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.sidebar.selectbox('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        parameter_n_jobs = st.sidebar.selectbox('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        # st.write(df.head(5))
        if st.button('Press to use Dataset'):

            with st.spinner(text='Loading dataset'):    
                df = json.load(uploaded_file)
                st.markdown('**1.1. Glimpse of dataset**')
                st.json(df)

            # Timer starts
            starttime = time.time()
            build_model(df)
            # Total time elapsed since the timer started
            totaltime = round((time.time() - starttime), 2)
            
            st.markdown('3. Benchmarking')
            st.success("Time taken = " + str(totaltime) + " seconds")
    else:
        st.info('Awaiting for JSON file to be uploaded.')
        # if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            #diabetes = load_diabetes()
            #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            #Y = pd.Series(diabetes.target, name='response')
            #df = pd.concat( [X,Y], axis=1 )

            #st.markdown('The Diabetes dataset is used as the example.')
            #st.write(df.head(5))

            # Boston housing dataset
            # boston = load_boston()
            # X = pd.DataFrame(boston.data, columns=boston.feature_names)
            # Y = pd.Series(boston.target, name='response')
            # df = pd.concat( [X,Y], axis=1 )

            # st.markdown('The Boston housing dataset is used as the example.')
            # st.write(df.head(5))

            # build_model(df)