import streamlit as st
import requests
from datetime import datetime, date
# from config import client_id, client_secret, api_key
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as df


def app():
    st.title('API Data Extractor')

    st.write('This is the `API Extractor` of the BCIT HVAC Machine Learning app.')

    st.write('In this app, we can extract the data using the Kaizen API.')

    st.title('Kaizen Data Pulling')
    st.write('We protect the privacy of the BCIT buildings by using the `Kaizen Building IDs` provided to us')

    @st.cache(ttl=(864000 - 10), persist=True, max_entries=1)
    def get_jwt():
        response = requests.post('https://login-global.coppertreeanalytics.com/oauth/token', data={
            'grant_type': 'client_credentials',
            'client_id': st.secrets.client_id,
            'client_secret': st.secrets.client_secret,
            'audience': 'organize'
        })
        return response.json()['access_token']

    jwt = get_jwt()

    building_id = st.text_input('Building ID', value=9871)

    if building_id:
        @st.cache(ttl=259200, persist=True, max_entries=20)
        def get_all_trend_logs():
            url = f'https://kaizen.coppertreeanalytics.com/yana/mongo/trend_log_summary/?building={building_id}&page=1&page_size=500'
            # else:
            trend_logs = []

            while url:
                response = requests.get(url, headers={
                    'Authorization': f'Bearer {jwt}'
                })

                body = response.json()
                trend_logs.extend(body['results'])
                url = body['links']['next']

            return trend_logs

        trend_logs = get_all_trend_logs()
        # x = np.where(trend_logs == 'NE01_AHU7_RESET_POLL_TL')

        # for dic_x in trend_logs:
        #     for dic_y in dic_x:
        #         for dic_z in dic_y:
        #             st.write(dic_z['name'])
        # for i in range(len(trend_logs)):
        #     x = np.where(trend_logs[i]["name"] == "NE01_AHU7_RESET_POLL_TL")
        #     st.write(i)
        #     st.write(trend_logs[i]["name"])
        default_selection = []
        if st.checkbox("Press to use example columns"):
            default_selection = [
                # NE01_AHU7_RESET_POLL_TL
                trend_logs[5684],
                # NE01_AHU7_HCV_POLL_TL
                trend_logs[5719],
                # NE01_AHU7_HC_SWT_POLL_TL
                trend_logs[5691],
                # NE01_AHU7_HC_RWT_POLL_TL
                trend_logs[5692],
                # NE01_AHU7_MAD_FB_POLL_TL
                trend_logs[5689],
                # NE01_AHU7_HC_SAT_POLL_TL
                trend_logs[5690],
                # NE01_AHU7_MAT_POLL_TL
                trend_logs[5687],
                # NE01_AHU7_RAT_POLL_TL
                trend_logs[5688],
                # NE01_AHU7_SF_SPD_POLL_TL
                trend_logs[5695],
                # NE01_AHU7_EF_SPD_POLL_TL
                trend_logs[5706],
                # NE01_AHU5_OAT_GV_POLL_TL
                trend_logs[6873],
                #  VAV 4-1 (Room 412) Actual Temperature (°C)
                trend_logs[5542],
                # VAV 4-2 (Room 411)
                trend_logs[5566],
                # VAV 4-3 (Room 410)
                trend_logs[5590],
                # VAV 4-4 (Room 409)
                trend_logs[5613],
                # VAV 4-5 (Room 408)
                trend_logs[5637],
                # VAV 4-6 (Room 407)
                trend_logs[5660],
                # VAV 4-7 (Room 415D)
                trend_logs[5726], ]
        else: 
            default_selection = []
        selected_trend_logs = st.multiselect('Columns to extract:', options=trend_logs,
                                             format_func=lambda x: x['name'], default = default_selection)

        if selected_trend_logs:
            if st.checkbox("See snapshot of data"):

                st.write(selected_trend_logs)
            window_start = st.date_input(
                'Data window start', value=date(2000, 1, 1))

            window_end = st.date_input(
                'Data window end', value=date(2020, 2, 1))

            window_start_str = datetime.fromordinal(
                window_start.toordinal()).isoformat()
            window_end_str = datetime.fromordinal(
                window_end.toordinal()).isoformat()

            if st.button(f'Pull Data'):
                def round_datetime(dt, minutes=15):
                    return dt.replace(minute=(dt.minute // minutes * minutes), second=0, microsecond=0)

                def request_column(trend_log_id: str, data_by_datetime: dict):
                    '''Get the data from one trend log and append its value indexed by `trend_log_id` to the dictionary `data_by_datetime` by its timestamp.'''

                    response = requests.get('https://kaizen.coppertreeanalytics.com/public_api/api/get_tl_data_start_end', params={
                        'api_key': st.secrets.api_key,
                        'tl': trend_log_id,
                        'start': window_start_str,
                        'end': window_end_str,
                        'format': 'json',
                        'data': 'raw'  # Don't round values to 2 decimal places
                    })

                    response_data = response.json()

                    for datum in response_data:
                        dt = round_datetime(
                            datetime.fromisoformat(datum['ts']))
                        if dt not in data_by_datetime:
                            data_by_datetime[dt] = {}
                        data_by_datetime[dt][trend_log_id] = datum['v']

                    return len(response_data)

                data_by_datetime = {}

                progress_bar = st.progress(0.0)

                for index, trend_log in enumerate(selected_trend_logs):
                    with st.spinner(trend_log['name']):
                        num_rows = request_column(
                            trend_log['_id'], data_by_datetime)
                    st.metric(label=trend_log['name'],
                              value=f'{num_rows} data points')
                    progress_bar.progress(
                        (index + 1) / len(selected_trend_logs))

                num_removed_by_reference = {}

                for dt, data in list(data_by_datetime.items()):
                    for trend_log in selected_trend_logs:
                        trend_log_id = trend_log['_id']
                        if trend_log_id not in data or data[trend_log_id] is None:
                            if trend_log_id not in num_removed_by_reference:
                                num_removed_by_reference[trend_log_id] = 0
                            num_removed_by_reference[trend_log_id] += 1
                            del data_by_datetime[dt]
                            break

                st.header('Summary of Removed Rows')
                for (data_reference, num_removed) in num_removed_by_reference.items():
                    st.metric(
                        label=f'Because {data_reference} was missing', value=num_removed)

                @st.cache
                def get_csv_string():
                    csv_string = ''
                    column_names = [x['name'] for x in selected_trend_logs]
                    csv_string += ','.join(['Timestamp'] + column_names) + '\n'
                    for dt, data in data_by_datetime.items():
                        columns = [dt.isoformat()] + [str(data[x['_id']])
                                                      for x in selected_trend_logs]
                        csv_string += ','.join(columns) + '\n'
                    return csv_string

                csv_string = get_csv_string()
                st.download_button(
                    label="Download data as CSV",
                    data=csv_string,
                    file_name='kaizen_data.csv',
                    mime='text/csv',
                )
                # corrmat = dt.corr()
                # f, ax = plt.subplots(figsize=(12, 9))
                # sns.heatmap(corrmat, cbar=True, annot=True,
                #             square=True, fmt='.2f')
                # st.write(f)
