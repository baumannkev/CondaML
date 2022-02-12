import streamlit as st
import hvacpredictor
import home
from multiapp import MultiApp
import hvacmodelbuilding # import your app modules here
import multioutputregressor
import modelmaker
from PIL import Image

st.set_page_config(page_title="BCIT HVAC", layout='wide', page_icon= Image.open('images/bcit.jpg'))

app = MultiApp()

st.markdown("""
# BCIT HVAC Machine Learning App
""")
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Model Maker", modelmaker.app)
app.add_app("HVAC Predictor", hvacpredictor.app)
# The main app
app.run()