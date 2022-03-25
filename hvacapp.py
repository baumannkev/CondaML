"""Entry point of the app

This app allows the user to create a model that predicts the temperature of specified rooms and creates a visual representation of the predicted output using graphs.

This app first takes in and accepts the API key for Kaizen from a TOML file to be able to access the building data using the API Extractor.
Afterwards, the user will input the building ID, the building data columns to be extracted, the date range of values to extract, and then assemble all data points into a single CSV file, which can be downloaded.
Finally, the user will use the provided CSV file to predict and display a graph of the output, as well as display some statistics about the input and output.
"""
import streamlit as st
from multiapp import MultiApp
# Import apps
import home
import apiextract
import modelling
from PIL import Image

st.set_page_config(page_title="BCIT HVAC", page_icon= Image.open('images/bcit.jpg'))

app = MultiApp()

st.markdown("""
# BCIT HVAC Machine Learning App
""")
# Add all your application here
app.add_app("Home", home.app)
app.add_app("API Extractor", apiextract.app)
app.add_app("Modelling", modelling.app)
# The main app
app.run()