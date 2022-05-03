"""API Extractor for the app

This script allows the user to access the Kaizen database using an API key from a TOML file (secrets.toml) and create and download a comma separated value file (kaizen_data.csv)
containing several columns' worth of data of a specific building from a range of dates.

If the user doesn't know what columns to extract, the script has default columns the user can use instead of choosing themselves.
"""
import streamlit as st
import requests
from datetime import datetime, date

# Check Input From User for Password
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["password"] == st.secrets["password"]):
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("""
            ## Login
        """)
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password", help="Please [contact](mailto:baumannkev@gmail.com) us to gain access to our API Extractor. We hold sensitive data from BCIT that we have decided only to share with a select few."
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.markdown("""
            ## Login
        """)
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password", help="Please [contact](mailto:baumannkev@gmail.com) us to gain access to our API Extractor. We hold sensitive data from BCIT that we have decided only to share with a select few."
        )
        st.error("😕 Password incorrect. Hover over the ❔ for more help")
        
        return False
    else:
        # Password correct.
        return True

def app():
    """Extracts data from database and produces a CSV file from specified inputs
    """

    st.caption("""
            <hr>
            """, unsafe_allow_html=True)
    # Check if password is correct
    if check_password():


        st.title('API Data Extractor')

        with st.expander("See explanation"):
            st.write("""
                This is the `API Extractor` of the BCIT HVAC Machine Learning app
                In this app, we can extract the data using the Kaizen API
            """)
            st.video("images/apiextractor.webm")

        st.title('Kaizen Data Pulling')
        st.write('We protect the privacy of the BCIT buildings by using the `Kaizen Building IDs` provided to us')

        @st.cache(ttl=(864000 - 10), persist=True, max_entries=1)
        def get_jwt():
            """Connects to the Kaizen database using the provided client ID

            Returns
            -------
            jwt
                a JSON Web Token containing encrypted data from the Kaizen database
            """
            response = requests.post('https://login-global.coppertreeanalytics.com/oauth/token', data={
                'grant_type': 'client_credentials',
                'client_id': st.secrets.client_id,
                'client_secret': st.secrets.client_secret,
                'audience': 'organize'
            })
            return response.json()['access_token']

        jwt = get_jwt()

        building_id = st.text_input(
            'Building ID', value=9871, help='The **building ID** is a numeric identifier of a building on Kaizen, which can be found at the end of the URL while viewing a building page.')

        if building_id:
            @st.cache(ttl=259200, persist=True, max_entries=20)
            def get_all_trend_logs():
                """Filters through the database using inputted building ID and produces a trend log

                Returns
                -------
                trend_logs : list
                    a list containing all trend logs stored in database for variable building_id
                """
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

            if st.checkbox('Use example columns'):
                default_names = [
                    'NE01_AHU7_RESET_POLL_TL',
                    'NE01_AHU7_HCV_POLL_TL',
                    'NE01_AHU7_HC_SWT_POLL_TL',
                    'NE01_AHU7_HC_RWT_POLL_TL',
                    'NE01_AHU7_MAD_FB_POLL_TL',
                    'NE01_AHU7_HC_SAT_POLL_TL',
                    'NE01_AHU7_MAT_POLL_TL',
                    'NE01_AHU7_RAT_POLL_TL',
                    'NE01_AHU7_SF_SPD_POLL_TL',
                    'NE01_AHU7_EF_SPD_POLL_TL',
                    'NE01_AHU5_OAT_GV_POLL_TL',
                    'VAV4_1_RT_TL',
                    'VAV4_2_RT_TL',
                    'VAV4_3_RT_TL',
                    'VAV4_4_RT_TL',
                    'VAV4_5_RT_TL',
                    'VAV4_6_RT_TL',
                    'VAV4_7_RT_TL'
                ]
                default_selection = [
                    x for x in trend_logs if x['name'] in default_names]
                default_selection.sort(key=lambda x: x['name'])
            else:
                default_selection = []

            selected_trend_logs = st.multiselect('Columns to extract', options=trend_logs,
                                                format_func=lambda x: x['name'], default=default_selection)

            if selected_trend_logs:
                # current datetime
                now = datetime.now()

                current_date = now.date()
                if st.checkbox('Show trend log details'):
                    st.write(selected_trend_logs)

                window_start = st.date_input(
                    'Data window start', value=date(2000, 1, 1))

                window_end = st.date_input(
                    'Data window end', value=current_date)

                window_start_str = datetime.fromordinal(
                    window_start.toordinal()).isoformat()
                window_end_str = datetime.fromordinal(
                    window_end.toordinal()).isoformat()

                if st.button(f'Pull Data'):
                    def round_datetime(dt, minutes=15):
                        """Takes in an inputted date and time and rounds it to the closest 15th minute

                        Parameters
                        ----------
                        dt
                            a datetime object to be rounded
                        minutes : optional
                            an int for the dt object to be compared to for rounding

                        Returns
                        -------
                            a datetime object rounded to the closest 15th minute
                        """
                        return dt.replace(minute=(dt.minute // minutes * minutes), second=0, microsecond=0)

                    def request_column(trend_log_id: str, data_by_datetime: dict):
                        """Get the data from one trend log and append its value indexed by `trend_log_id` to the dictionary `data_by_datetime` by its timestamp.

                        Parameters
                        ----------
                        trend_log_id : str
                            a string containing the ID of the trend log to be extracted from the database
                        data_by_datetime : dict
                            a dictionary containing the timestamp

                        Returns
                        -------
                        int
                            the number of items retrieved from database using specified trend log ID, start timestamp, and end timestamp"""

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

                    st.balloons()

                    @st.cache
                    def get_csv_string():
                        """Produces a CSV string consisting of all gathered data from database using specified, inputted values

                        Returns
                        -------
                        csv_string : str
                            a string containing gathered data separated by commas
                            """
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
