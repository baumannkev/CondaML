import streamlit as st
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# st.title('Data Modelling')

# st.header('Data Selection')

def app():
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing some data", type='csv')

    if uploaded_file is not None:
        @st.cache(allow_output_mutation=True, max_entries=5)
        def get_dataframe():
            dataframe = pd.read_csv(uploaded_file)
            dataframe.dropna(inplace=True)
            return dataframe

        dataframe = get_dataframe().copy()

        all_column_names = list(dataframe.columns)

        input_column_names = st.multiselect('Inputs', all_column_names)
        output_column_names = st.multiselect('Outputs', all_column_names)

        st.caption(
            'Select which columns of the data will be inputs and which will be predicted outputs.')

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

            if st.checkbox('Preview all data'):
                st.dataframe(dataframe)

            st.header('Model Setup')

            test_size = st.slider('Test data percent',
                                min_value=0.01, max_value=0.99, value=0.2,
                                help='The fraction of the data reserved for testing instead of training the model.'
                                )
            train_dataframe, test_dataframe = train_test_split(
                dataframe, test_size=test_size, shuffle=False, random_state=2022_02_02)

            if st.checkbox('Preview train/test data'):
                st.write('Train data')
                st.dataframe(train_dataframe)
                st.write('Test data')
                st.dataframe(test_dataframe)

            train_input_values = train_dataframe[input_column_names].values
            train_output_values = train_dataframe[output_column_names].values

            @st.cache(allow_output_mutation=True, show_spinner=False)
            def get_pipeline():
                pipeline = make_pipeline(
                    StandardScaler(),
                    ExtraTreesRegressor(n_estimators=100, random_state=0)
                )
                pipeline.fit(train_input_values,
                            train_output_values)
                return pipeline

            with st.spinner('Fitting model...'):
                pipeline = get_pipeline()

            st.success('Model is ready!')

            @st.cache(max_entries=100)
            def get_model_mean_absolute_errors():
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
                    test_prediction_columns = list(zip(*test_prediction_outputs))

                    for i in range(len(output_column_names)):
                        output_mean_absolute_errors.append(mean_absolute_error(
                            test_output_columns[i], test_prediction_columns[i]))

                return (total_mean_absolute_error, output_mean_absolute_errors)

            total_mean_absolute_error, output_mean_absolute_errors = get_model_mean_absolute_errors()

            st.header('Model Testing')

            st.subheader('Mean Absolute Errors')

            st.metric(label='Mean of all outputs',
                    value=total_mean_absolute_error)

            if len(output_column_names) > 1:
                for i, column_name in enumerate(output_column_names):
                    st.metric(label=column_name,
                            value=output_mean_absolute_errors[i])

            st.header('Prediction')

            col1, col2 = st.columns(2)

            col1.subheader('Inputs')

            prediction_inputs = []
            default_inputs = test_dataframe.values[0].tolist()

            for index, column_name in enumerate(input_column_names):
                prediction_inputs.append(col1.slider(column_name, min_value=-100.0,
                                                    max_value=100.0, value=default_inputs[index]))

            col2.subheader('Predicted Outputs')

            predictions = pipeline.predict([prediction_inputs])

            st.write(type(predictions))

            if len(output_column_names) > 1:
                [prediction_outputs] = predictions
            else:
                prediction_outputs = predictions

            for index, predicted_value in enumerate(prediction_outputs):
                column_name = output_column_names[index]
                col2.metric(label=column_name, value=predicted_value)