import streamlit as st
import pickle
future_time = 1
size_window = 1


# loading in the model to predict on the data
# pickle_in = open('ExtraTreesRegressor.pkl', 'rb')
pickle_in = open('SGDRegressor.pkl', 'rb')
extraTreesRegressor = pickle.load(pickle_in)
pickle_in1 = open('SGDRegressor.pkl', 'rb')
sgdRegressor = pickle.load(pickle_in1)


def prediction(cool_coil_valve, hot_water_valve, hot_water_supply_temp, hot_water_return_temp,
               dampers_pos, supply_air_temp, mixed_air_temp, return_air_temp, supply_fan_speed, return_fan_speed,
               outside_air_temp, option):
    if option == "ExtraTreesRegressor":
        prediction = extraTreesRegressor.predict([[cool_coil_valve, hot_water_valve, hot_water_supply_temp, hot_water_return_temp,
                                        dampers_pos, supply_air_temp, mixed_air_temp, return_air_temp, supply_fan_speed, return_fan_speed,
                                        outside_air_temp]])
    elif option == "SGDRegressor":
        prediction = sgdRegressor.predict([[cool_coil_valve, hot_water_valve, hot_water_supply_temp, hot_water_return_temp,
                                dampers_pos, supply_air_temp, mixed_air_temp, return_air_temp, supply_fan_speed, return_fan_speed,
                                outside_air_temp]])
    # col1 = st.columns([3, 1])
    # col1.subheader("Here is your prediction results!")
    col1 , col2= st.columns([5, 5])
    col1.subheader("Prediction Results")
    import numpy as np
    import pandas as pd

    numChange = 0
    
    dict={}
    for i in range(0, len(prediction)):
        # column_name = df['columns'][1][i][1]
        pred = prediction[i]
        dict[i] = prediction[i]
        # st.metric(f'test', str(pred) + '°C')
    col1.write(pred)
    st.write(dict)

    from bokeh.plotting import figure

    x = [1, 2, 3, 4, 5, 6, 7]
    y = pred

    p = figure(
        title='Prediction Trend',
        x_axis_label='Rooms',
        y_axis_label='Temperature')

    p.line(x, y, legend_label='Trend', line_width=3)

    st.bokeh_chart(p, use_container_width=True)
    # chart_data = pd.DataFrame(
    #     data=pred,
    #     columns=['a'])
    # st.area_chart(chart_data)
    chart_data = pd.DataFrame(
            data=dict)
    st.subheader('Bar Chart')
    st.bar_chart(chart_data)

    st.subheader('Line Chart')
    st.line_chart(chart_data)

    st.subheader('Area Chart')
    st.area_chart(chart_data)

def app():
    option = st.sidebar.selectbox('Select ML Algorithm', ('SGDRegressor','ExtraTreesRegressor',
                                                  'Next Model', ))
    st.write("""
    # BCIT Room Temperature Prediction App

    This app predicts the **room temperatures** at BCIT!

    Data obtained from Kaizen CopperTreeAnalytics
    """)

    st.sidebar.header('User Input Features')

    room = st.sidebar.selectbox('Room', ('Room 412', 'Room 411',
                                'Room 410', 'Room 409', 'Room 408', 'Room 407', 'Room 415D'))

    st.sidebar.header('Prediction Parameters')
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
        st.write('Try adjusting the parameters')
        prediction(cool_coil_valve, hot_water_valve, hot_water_supply_temp, hot_water_return_temp,
                   dampers_pos, supply_air_temp, mixed_air_temp, return_air_temp, supply_fan_speed, return_fan_speed,
                   outside_air_temp, option)

    else:
        st.write("Adjust parameters")
