import streamlit as st
import hvacmlapp
import home
from multiapp import MultiApp
import hvacmodelbuilding # import your app modules here

app = MultiApp()

st.markdown("""
# BCIT HVAC Machine Learning App
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Machine Learning App", hvacmlapp.app)
app.add_app("Machine Learning Model Builder", hvacmodelbuilding.app)
# The main app
app.run()