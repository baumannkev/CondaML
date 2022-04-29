import streamlit as st
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


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
        input_column_names = st.multiselect(
            'Inputs', all_column_names, default=default_input_columns)
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
                ('Extra-Trees', 'SGD'))

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
                """Creates a pipeline using ExtraTreesRegressor model and returns it to be used for the modelling.

                Returns
                -------
                pipeline : Pipeline
                    a pipeline object made from ExtraTreesRegressor using values and inputs from app"""
                print(model_type)
                if model_type == 'Extra-Trees':
                    regressor = ExtraTreesRegressor(
                        n_estimators=100, random_state=0)
                else:
                    regressor = MultiOutputRegressor(
                        SGDRegressor(max_iter=1_000_000, tol=1e-6))
                pipeline = make_pipeline(
                    StandardScaler(),
                    regressor
                )
                pipeline.fit(train_input_values,
                             train_output_values)
                return pipeline

            with st.spinner('Fitting model...'):
                pipeline = get_pipeline()

            st.success('Model is ready!')

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

            st.caption("""
            The mean absolute error (MAE) takes the absolute difference between the actual and forecasted values and finds the average.
            
            Here we can see the MAE of each output and its delta with the total mean of all the outputs.
            """)
            st.subheader('Mean Absolute Errors')
            st.metric(label='Mean of all outputs',
                      value="{}".format(total_mean_absolute_error))

            if len(output_column_names) > 1:
                for i, column_name in enumerate(output_column_names):
                    st.metric(label=column_name,
                              value="{}".format(output_mean_absolute_errors[i]), delta=(output_mean_absolute_errors[i] - total_mean_absolute_error))

            st.caption("""
            <hr>
            """, unsafe_allow_html=True)

            st.header('Prediction')

            with st.expander("See Explanation"):
                st.write("""
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
                col2.metric(label=column_name, value="{}".format(predicted_value), delta="{}".format(
                    predicted_value - st.session_state[str(index)]))

            st.bar_chart(prediction_outputs)
