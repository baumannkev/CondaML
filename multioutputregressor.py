from math import sqrt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
future_time = 1
size_window = 1


def app():

    #---------------------------------#
    def plot_predictions(test, predicted, title):
        fig = plt.figure(figsize=(32, 8))
        plt.title(title)
        plt.plot(test, color='blue', label='Actual Supply Air Temperature')
        plt.plot(predicted, color='red',
                 label='Predicted Supply Air Temperature')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        st.balloons()
        st.pyplot(fig)

    # Model building
    def build_model(df):

        # Reserve 20% of the data for testing

        with st.spinner(text='Training model'):
            train_rows, test_rows = train_test_split(
                df['rows'], test_size=0.2, shuffle=True, random_state=2022_02_02)

            st.markdown('**1.2. Data splits**')
            st.write('Training set')
            st.success(f'Number of training data: {len(train_rows)}')
            st.write('Test set')
            st.success(f'Number of testing data: {len(test_rows)}')

            # Both are tuples of lists of values
            train_input_points, train_output_points = list(zip(*train_rows))

            # Create Model

        with st.spinner(text='Creating model'):
            pipeline = make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(SGDRegressor(max_iter=parameter_max_iter,
                                                  tol=1e-6,
                                                  shuffle=parameter_shuffle,

                                                  )
                                     )
            )
            # Fit model to training datasets
            pipeline.fit(train_input_points, train_output_points)

        # Test model_selection
        st.subheader('2. Model Testing')
        with st.spinner(text='Testing model_selection'):
            all_mean_absolute_errors = []
            all_mean_squared_errors = []

            for [inputs, outputs] in test_rows:
                [prediction] = pipeline.predict([inputs])

                absolute_errors = [abs(p - o)
                                   for p, o in zip(prediction, outputs)]
                squared_errors = [(p-o)*(p-o)
                                  for p, o in zip(prediction, outputs)]

                mean_absolute_error = sum(
                    absolute_errors) / len(absolute_errors)
                mean_squared_error = sum(squared_errors) / len(squared_errors)

                all_mean_absolute_errors.append(mean_absolute_error)
                all_mean_squared_errors.append(mean_squared_error)

        st.markdown('**2.1. Training set**')
        mean_mean_absolute_error = sum(
            all_mean_absolute_errors) / len(all_mean_absolute_errors)
        root_mean_mean_squared_error = sqrt(
            sum(all_mean_squared_errors) / len(all_mean_squared_errors))

        st.write('Error (MSE or MAE):')

        st.success(f'Mean absolute error: {mean_mean_absolute_error} °C')
        st.success(f'Root mean squared error: {root_mean_mean_squared_error}')

        # Prediction Example
        st.markdown('**2.2. Prediction Example**')

    
               # st.info("Inputs")
            # col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(
            #     11)
            # col1.metric("Cooling Coil Valve (% open)", "37.6130981 %", "1.2 °F")
            #col2.metric("Hot Water Valve (% open)", "0.0 %", "-8%")
            # col3.metric("Hot Water Supply Temperature (°C)", "19.9593983 °C", "4%")
            # col4.metric("Hot Water Return Temperature (°C)", "20.8191605 °C", "4%")
            #col5.metric("Dampers Position (% open)", "20.4688644 %", "4%")
            # col6.metric("Supply Air Temperature (°C)", "19.946228 °C", "4%")
            # col7.metric("Mixed Air Temperature (°C)", "20.8940868 °C", "4%")
            # col8.metric("Return Air Temperature (°C)", "21.5931683 °C", "4%")
            # col9.metric("Supply Fan Speed (% of max speed)", "65.0570679 °C", "4%")

            # col10.metric("Return Fan Speed (% of max speed)", "53.0570641 %", "4%")

            # col11.metric("Outside Air Temperature (°C)", "19.1146049%", "4%")

        # if st.button('Predict'):
        [prediction] = pipeline.predict([[
            cool_coil_valve,
            hot_water_valve,
            hot_water_supply_temp,
            hot_water_return_temp,
            dampers_pos,
            supply_air_temp,
            mixed_air_temp,
            return_air_temp,
            supply_fan_speed,
            return_fan_speed,
            outside_air_temp
        ]])

        for i in range(0, len(prediction)):
            column_name = df['columns'][1][i][1]
            st.info(f'{column_name} -> {prediction[i]} °C')

             

    #---------------------------------#
    st.write("""
    # BCIT Room Temperature Machine Learning Model Builder App using Multioutput Regressor
    In this implementation, the *Multioutput Regressor()* function is used in this app for build a regression model using the **SGDRegressor** algorithm.
    Try adjusting the hyperparameters!
    """)

    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your JSON data'):
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input JSON file", type=["JSON"])

    # Sidebar - Specify parameter settings
    # with st.sidebar.header('2. Set Parameters'):
    st.sidebar.header('2. Set Parameters')

    with st.sidebar.subheader('2.1. Learning Parameters'):
        parameter_max_iter = st.sidebar.slider('Max Iterations (max_iter)', 0, 1_000_000, 1_000_000, 1000, help=(
            "The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method."))
        parameter_shuffle = st.sidebar.selectbox(
            'Shuffle (Whether or not the training data should be shuffled after each epoch.)', options=[True, False])

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
            'Mixed Air Temperature (°C)', 0.0, 100.0,20.8940868 )  # 7
        return_air_temp = st.sidebar.slider(
            'Return Air Temperature (°C)', 0.0, 100.0, 21.5931683 )  # 8
        supply_fan_speed = st.sidebar.slider(
            'Supply Fan Speed (% of max speed)', 0.0, 100.0, 65.0570679)  # 9
        return_fan_speed = st.sidebar.slider(
            'Return Fan Speed (% of max speed)', 0.0, 100.0, 53.0570641)  # 10
        outside_air_temp = st.sidebar.slider(
            'Outside Air Temperature (°C)', 0.0, 100.0, 19.1146049)  # 11

        # parameter_n_estimators = st.sidebar.slider(
        #     'Number of estimators (n_estimators)', 0, 1000, 100, 100)
        # parameter_max_features = st.sidebar.select_slider(
        #     'Max features (max_features)', options=['auto', 'sqrt', 'log2'])
        # parameter_min_samples_split = st.sidebar.slider(
        #     'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        # parameter_min_samples_leaf = st.sidebar.slider(
        #     'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    # with st.sidebar.subheader('2.2. General Parameters'):
    #     parameter_random_state = st.sidebar.slider(
    #         'Seed number (random_state)', 0, 1000, 42, 1)
    #     parameter_criterion = st.sidebar.selectbox(
    #         'Performance measure (criterion)', options=['mse', 'mae'])
    #     parameter_bootstrap = st.sidebar.selectbox(
    #         'Bootstrap samples when building trees (bootstrap)', options=[True, False])
    #     parameter_oob_score = st.sidebar.selectbox(
    #         'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    #     parameter_n_jobs = st.sidebar.selectbox(
    #         'Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    #---------------------------------#
    # Main panel

    

    # Displays the dataset
    st.subheader('1. Dataset')

    # st.button('Press to predict', on_click= showBar())
        
    if uploaded_file is not None:
        # st.write(df.head(5))
        if st.button('Press to use Dataset'):
            st.write('Try adjusting the hyperparameters')

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

       
