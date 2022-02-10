import streamlit as st
import hvacmlapp
import home
from multiapp import MultiApp
import hvacmodelbuilding # import your app modules here
import multioutputregressor
app = MultiApp()

st.markdown("""
# BCIT HVAC Machine Learning App
""")
primary_clr = st.get_option("theme.primaryColor")
txt_clr = st.get_option("theme.textColor")
# I want 3 colours to graph, so this is a red that matches the theme:
# primary_clr = st.get_option("theme.primaryColor")
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Machine Learning App", hvacmlapp.app)
app.add_app("Machine Learning Model Builder", hvacmodelbuilding.app)
app.add_app("Multioutput Regressor Model", multioutputregressor.app)
# The main app
app.run()