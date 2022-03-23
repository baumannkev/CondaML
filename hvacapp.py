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