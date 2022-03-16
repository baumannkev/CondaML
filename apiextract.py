import streamlit as st
import requests
from datetime import datetime, date
# from config import client_id, client_secret, api_key

def app():
    st.title('API Data Extractor')

    st.write('This is the `API Extractor` of the BCIT HVAC Machine Learning app.')

    st.write('In this app, we can extract the data using the Kaizen API.')


    st.title('Kaizen Data Pulling')


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

        selected_trend_logs = st.multiselect('Columns to extract:', options=trend_logs,
                                            format_func=lambda x: x['name'])

        if selected_trend_logs:
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
                        dt = round_datetime(datetime.fromisoformat(datum['ts']))
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
                    progress_bar.progress((index + 1) / len(selected_trend_logs))

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