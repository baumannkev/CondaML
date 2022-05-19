import streamlit as st
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time



def app():
    """Produces a model from an uploaded CSV file that came from the API Extractor.

    If no CSV file is uploaded, the user can use an example dataset automatically provided in the dataset folder."""

    st.caption("""
            <hr>
            """, unsafe_allow_html=True)

    st.title('Data Modelling')

    with st.expander("See Explanation"):
        st.write("""
            This modelling app allows the user to create a model that predicts the temperature of specified rooms from an uploaded CSV file and creates a visual representation of the predicted output using graphs

            The model creates a pipeline using ExtraTreesRegressor Machine Learning algorithm from the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) library and returns it to be used for the modelling 

        """)
        st.video("images/modelling.webm")
    st.header('Data Selection')

    st.caption(
        'Choose to use an Example Dataset extracted from the API Extractor. The input and output columns will be autofilled but still editable.')
    user_example = False
    uploaded_file = ""
    if st.checkbox('Use example dataset'):
        uploaded_file = open("dataset/example_dataset.csv")
        user_example = True
    else:
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing some data", type='csv')

    if uploaded_file is not None:
        @st.cache(allow_output_mutation=True, max_entries=5)
        def get_dataframe():
            """Reads the uploaded CSV file and provides a dataframe to be used for modelling.

            Returns
            -------
            dataframe : DataFrame
                a dataframe object consisting of data extracted from the uploaded CSV file"""
            dataframe = pd.read_csv(uploaded_file)
            dataframe.dropna(inplace=True)
            return dataframe

        dataframe = get_dataframe().copy()

        dataFrame2 = get_dataframe().copy()

        all_column_names = list(dataframe.columns)

        # Append the items from the example dataset to the default inputs and outputs
        if user_example:
            default_input_columns = all_column_names[1:-7]
            default_output_columns = all_column_names[-7:-1]
        else:
            default_input_columns = []
            default_output_columns = []

        st.caption(
            'Select which columns of the data will be inputs and which will be predicted outputs.')

        # If the 'Use example dataset is unchecked, we give the option to select all the input columns.
        if (user_example == False):
             add_all_input_columns = st.checkbox("Add all inputs")
             if add_all_input_columns:
                 default_input_columns = all_column_names
        input_column_names = st.multiselect(
            'Inputs', all_column_names, default=default_input_columns)
        if (user_example == False):
             add_all_output_columns = st.checkbox("Add all outputs")
             if add_all_output_columns:
                 default_output_columns = all_column_names
        output_column_names = st.multiselect(
            'Outputs', all_column_names, default=default_output_columns)

        for column_name in input_column_names:
            if column_name in output_column_names:
                st.warning(f'**{column_name}** is in both inputs and outputs!')
        if input_column_names and output_column_names:

            dataframe.drop(columns=[
                x for x in dataframe.columns if x not in input_column_names and x not in output_column_names], inplace=True)

            shift_amount = st.slider(
                'Output shift amount', min_value=-10, max_value=10, value=1,
                help='For example, if a row represents a period of **15 minutes**, an output shift of **1** will make the outputs represent the data **15 minutes after the inputs**.'
            )

            dataframe[output_column_names] = dataframe[output_column_names].shift(
                shift_amount)
            dataframe.dropna(inplace=True)

            st.header('Model Setup')

            test_size = st.slider('Test data percent',
                                  min_value=0.01, max_value=0.99, value=0.2,
                                  help='The fraction of the data reserved for testing instead of training the model.'
                                  )
            train_dataframe, test_dataframe = train_test_split(
                dataframe, test_size=test_size, shuffle=False, random_state=2022_02_02)

            model_type = st.selectbox(
                'Model Type',
                 ('Extra-Trees', 'SGD', 'GBR', 'XGB'))
            # Global parameters 
            parameter_n_estimators = 100
            parameter_random_state = 0

            # Extra trees parameters
            parameter_max_features = 1.0
            parameter_min_samples_split = 2
            parameter_min_samples_leaf = 1
            parameter_criterion = "squared_error"
            parameter_bootstrap = False
            parameter_oob_score = False
            parameter_n_jobs = 1
            
            # SGD parameters
            parameter_tol = 1e-3
            parameter_max_iter = 1000
            parameter_loss = 'squared_error'

            if model_type == 'Extra-Trees':
                st.info("ExtraTreesRegressor - fits a number of randomized decision (extra) trees on various subsamples of the dataset and uses averaging to improve predictive accuracy and control over-fitting")
            elif model_type == 'SGD':
                st.info("Stochastic Gradient Descent (SGD) - the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (learning rate)")
            elif model_type == 'GBR':
                st.info("GradientBoostinRegressor (GBR) - builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.")
            elif model_type == 'XGB':
                st.info("XGBoost (XGB) - is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. ")

            if st.checkbox('Modify Model Parameters', help="Check this box to open the Model parameters sidebar"):

                st.sidebar.markdown(model_type + " Model")
                with st.sidebar.subheader('1.0. Learning Parameters'):
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
                    if model_type == 'Extra-Trees':
                        parameter_max_features = st.sidebar.selectbox('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
                        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
                        parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

                with st.sidebar.subheader('2.0. General Parameters'):
                    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 0, 1)
                    if model_type == 'Extra-Trees':
                        parameter_criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['squared_error', 'absolute_error'])
                        parameter_bootstrap = st.sidebar.selectbox('Bootstrap samples when building trees (bootstrap)', options=[False, True])
                        parameter_oob_score = st.sidebar.selectbox('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
                        parameter_n_jobs = st.sidebar.selectbox('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
                    if model_type == 'SGD':
                        # parameter_tol = st.sidebar.slider('Tol', 1e-3, 1e-0, 1e-6, 1e-10)
                        parameter_max_iter = st.sidebar.slider('Max Iterations (max_iter)', 1000, 1_000_000, 1000, 100)
                        parameter_loss = st.sidebar.selectbox('Loss function', options=['squared_error', "huber", "epsilon_insensitive"], help="The loss function to be used. The possible values are squared_error, huber, epsilon_insensitive or ‘squared_epsilon_insensitive.The squared_error refers to the ordinary least squares fit. huber modifies squared_error to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon.‘epsilon_insensitive ignores errors less than epsilon and is linear past that; this is the loss function used in SVR. squared_epsilon_insensitive is the same but becomes squared loss past a tolerance of epsilon.")
                    if model_type == 'GBR':
                        parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
                        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
                        parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
                        parameter_loss = st.sidebar.selectbox('Loss function', options=['squared_error', "huber", "absolute_error", "quantile"], help="The loss function to be used. The possible values are squared_error, huber, epsilon_insensitive or ‘squared_epsilon_insensitive.The squared_error refers to the ordinary least squares fit. huber modifies squared_error to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon.‘epsilon_insensitive ignores errors less than epsilon and is linear past that; this is the loss function used in SVR. squared_epsilon_insensitive is the same but becomes squared loss past a tolerance of epsilon.")
            
            if st.checkbox('Visualize data'):
                st.header('All data')
                st.info("Details of the data")
                st.write(dataframe.describe())
                testData = dataframe.describe().drop("count")
                st.area_chart(testData, use_container_width=True)
                st.info("All data in table")
                st.dataframe(dataFrame2)
                st.info("Heatmap of the dataframe")
                correlations = dataframe.corr()
                fig, ax = plt.subplots(figsize=(12, 9))
                sns.heatmap(correlations, cbar=True, annot=True,
                            square=True, fmt='.2f')
                st.pyplot(fig)
                if st.checkbox('View Histograms'):
                    st.header('Histograms')
                    for column_name in input_column_names + output_column_names:
                        st.subheader(column_name)
                        fig, ax = plt.subplots()
                        sns.histplot(dataframe[column_name], cbar=True)
                        st.pyplot(fig)

            train_input_values = train_dataframe[input_column_names].values
            train_output_values = train_dataframe[output_column_names].values

            @st.cache(allow_output_mutation=True, show_spinner=False, max_entries=5)
            def get_pipeline():
                """Creates a pipeline using the selected model and returns it to be used for the modelling. Models include ExtraTreesRegressor, SGDRegressor, and GradientBoostingRegressor.

                ExtraTreesRegressor - fits a number of randomized decision (extra) trees on various subsamples of the dataset and uses averaging to improve predictive accuracy and control over-fitting
                - Current Parameters:
                    - n_estimators - number of trees (set as 100)
                    - random_state - controls three sources of randomness (set as 0)
                - Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

                SGDRegressor - Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (learning rate)
                - Current Parameters:
                    - max_iter - maximum number of iterations over the training data (epochs) (set as 1,000,000)
                    - tol - the stopping criterion (set as 1e-6 or 0.000001)
                - Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html

                GradientBoostinRegressor - builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.
                - Source https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

                XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. 
                - Source https://github.com/dmlc/xgboost


                Returns
                -------
                pipeline : Pipeline
                    a pipeline object made from the selected model using values and inputs from app"""
                
                print(model_type)
                if model_type == 'Extra-Trees':
                    regressor = ExtraTreesRegressor(
                        n_estimators=parameter_n_estimators,
                        random_state=parameter_random_state,
                        max_features=parameter_max_features,
                        criterion=parameter_criterion,
                        min_samples_split=parameter_min_samples_split,
                        min_samples_leaf=parameter_min_samples_leaf,
                        bootstrap=parameter_bootstrap,
                        oob_score=parameter_oob_score,
                        n_jobs=parameter_n_jobs)
                elif model_type == 'GBR':
                    regressor = MultiOutputRegressor(
                        GradientBoostingRegressor(
                            n_estimators=parameter_n_estimators,
                            random_state=parameter_random_state,
                            max_features=parameter_max_features,
                            min_samples_split=parameter_min_samples_split,
                            min_samples_leaf=parameter_min_samples_leaf,
                            loss= parameter_loss))
                elif model_type == 'XGB':
                    regressor = MultiOutputRegressor(
                        xgb.XGBRegressor(
                            n_estimators=parameter_n_estimators,
                            random_state=parameter_random_state,
                            ))
                else:
                    regressor = MultiOutputRegressor(
                        SGDRegressor(
                            random_state=parameter_random_state,
                            max_iter=parameter_max_iter,
                            tol=parameter_tol,
                            loss= parameter_loss))
                pipeline = make_pipeline(
                    StandardScaler(),
                    regressor
                )
                pipeline.fit(train_input_values,
                             train_output_values)
                return pipeline

            with st.spinner('Fitting model...'):
                # Timer starts
                startTime = time.time()
                pipeline = get_pipeline()
                 # Total time elapsed since the timer started
                totalTime = round((time.time() - startTime), 2)

            st.success('Model is ready! Time taken: ' + str(totalTime) + 's')

            @st.cache(max_entries=100)
            def get_model_mean_absolute_errors(model_type):
                """Calculates and returns the total and output mean absolute errors of the model.

                Returns
                -------
                (total_mean_absolute_error, output_mean_absolute_error)
                    a tuple containing the total and output mean absolute error of the created model"""
                test_input_values = test_dataframe[input_column_names].values

                test_output_values = test_dataframe[output_column_names].values
                if len(output_column_names) == 1:
                    test_output_values = [x[0] for x in test_output_values]

                test_prediction_outputs = pipeline.predict(test_input_values)

                total_mean_absolute_error = mean_absolute_error(
                    test_output_values, test_prediction_outputs)
                output_mean_absolute_errors = []

                if len(output_column_names) > 1:
                    test_output_columns = list(zip(*test_output_values))
                    test_prediction_columns = list(
                        zip(*test_prediction_outputs))

                    for i in range(len(output_column_names)):
                        output_mean_absolute_errors.append(mean_absolute_error(
                            test_output_columns[i], test_prediction_columns[i]))

                return (total_mean_absolute_error, output_mean_absolute_errors)

            total_mean_absolute_error, output_mean_absolute_errors = get_model_mean_absolute_errors(
                model_type)

            st.header('Model Testing')

            st.info("""
            The mean absolute error (MAE) takes the absolute difference between the actual and forecasted values and finds the average.
            
            Here we can see the MAE of each output and its delta with the total mean of all the outputs.
            """)
            st.subheader('Mean Absolute Errors')
            st.metric(label='Mean of all outputs',
                      value="{}".format(total_mean_absolute_error))

            if len(output_column_names) > 1:
                for i, column_name in enumerate(output_column_names):
                    st.metric(label=column_name,
                              value="{}".format(output_mean_absolute_errors[i]), delta=( total_mean_absolute_error - output_mean_absolute_errors[i]))

            st.caption("""
            <hr>
            """, unsafe_allow_html=True)

            st.header('Prediction')

            st.info("""
                The predicted outputs shows the rooms, temperature and the delta.
                
                The values updates real-time when input sliders are modified.
            """)

            col1, col2 = st.columns(2)

            col1.subheader('Inputs')

            prediction_inputs = []
            default_inputs = test_dataframe.values[0].tolist()

            for index, column_name in enumerate(input_column_names):
                min_value = dataframe[column_name].min() - 10
                max_value = dataframe[column_name].max() + 10
                prediction_inputs.append(col1.slider(column_name, min_value=min_value,
                                                     max_value=max_value, value=default_inputs[index]))

            col2.subheader('Predicted Outputs')

            predictions = pipeline.predict([prediction_inputs])

            "data", st.session_state

            if len(output_column_names) > 1:
                [prediction_outputs] = predictions
            else:
                prediction_outputs = predictions

            for (i, item) in enumerate(prediction_outputs):
                if str(i) not in st.session_state:
                    st.session_state[i] = item

            i = 0
            for index, predicted_value in enumerate(prediction_outputs):

                column_name = output_column_names[index]
                # Delta is change compared to previous prediction output
                col2.metric(label=column_name, value="{}".format(predicted_value), delta="{}".format(
                    predicted_value - st.session_state[str(index)]))

            # st.bar_chart(prediction_outputs)
