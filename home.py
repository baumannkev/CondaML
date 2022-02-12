import streamlit as st
import altair as alt
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime

from utils import chart, db


def app():
    st.title('Home')

    st.write('This is the `home page` of the BCIT HVAC Machine Learning app.')

    st.write(
        'In this app, we will be building a simple regression model using the Trend Logs dataset.')

    COMMENT_TEMPLATE_MD = """{} - {}
    > {}"""

    # Comments part

    def space(num_lines=1):
        """Adds empty lines to the Streamlit app."""
        for _ in range(num_lines):
            st.write("")

    conn = db.connect()
    comments = db.collect(conn)

    with st.expander("💬 Open comments"):

        # Show comments

        st.write("**Comments:**")

        for index, entry in enumerate(comments.itertuples()):
            st.markdown(COMMENT_TEMPLATE_MD.format(
                entry.name, entry.date, entry.comment))

            is_last = index == len(comments) - 1
            is_new = "just_posted" in st.session_state and is_last
            if is_new:
                st.success("☝️ Your comment was successfully posted.")

        space(2)

        # Insert comment

        st.write("**Add your own comment:**")
        form = st.form("comment")
        name = form.text_input("Name")
        comment = form.text_area("Comment")
        submit = form.form_submit_button("Add comment")

        if submit:
            date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            db.insert(conn, [[name, comment, date]])
            if "just_posted" not in st.session_state:
                st.session_state["just_posted"] = True
            st.experimental_rerun()
