import streamlit as st
import altair as alt
import pandas as pd
from streamlit_disqus import st_disqus
import scipy


def app():
    """Displays and prints the homepage of the app, including welcome tag, instructions, and information about app.
    """
    
    st.caption("""
            <hr>
            """, unsafe_allow_html=True)

    st.title('Home')
    st.write('Welcome to the `BCIT HVAC Machine Learning app`')
    st.write(
        'To get started, click on the `Home` button on the navigation bar above, and navigate to our API Extractor.')

    st.image('images/bcitbuilding.jpg',
             caption="British Columbia Institute of Technology")

    st.markdown(
        '<style>img {border: 5px solid #fff400;}</style>', unsafe_allow_html=True)

    st.info('To learn more about our app, keep reading.')

    st.markdown('''
    # Our Problem
    In our vast energy dependant world, we recognize the problem of energy waste and cost that plagues the energy sector.
    
    For example, `30-45%` of a building's energy use is devoted to heating & cooling alone.

    In addition to that, `30%` of energy in buildings is wasted.

    ### What are the consequences of this?

    `Poor Management` and maintanance of HVAC systems lead to:
    * Significant energy waste
    * Reduces the comfort of the occupants 
    * A threat to our environment
    ''')

    st.markdown('''
    # Solution
    Through the application of Machine Learning Algorithms in BCIT's Building Management System (BMS) we can utilize:
    * Internet of Things (IoT) and Machine Learning Development to teach machines how to learn the patterns of past data
    * Deep learning & Neural Networks to predict system features
    ''')

    st.markdown('''
    All this is possible through the availability of reliable data at BCIT.
    As well as the complexity of the BCIT management system and the need for Machine Learning analysis. 
    ''')
    st.image('images/BCIT-HVAC.png', caption='BCIT BMS')

    st.markdown('''
    # Methodology
    Our proposed research led us to utilize the historical available data from BCIT, and to gain insights from the Facilities team.
    We developed this app using Python and the Streamlit library.
    
    The BCIT Computer Systems Technology COMP 3800 Team in collaboration with the BCIT Mechanical Engineering Cap Stone Team have developed this BCIT HVAC System Machine Learning Model and Predictor. 
    ''')

    st.markdown('''
    # Our next steps
    Our next steps include: 
    * Incorporate Deep Learning to the model
    * Forecast data in a more precise and quick manner
    * Implement optimization to BCIT's HVAC System 
    ''')

    def space(num_lines=1):
        """Adds empty lines to the Streamlit app."""
        for _ in range(num_lines):
            st.write("")
    st.markdown("[Scroll up](#bcit-hvac-machine-learning-app)",
                unsafe_allow_html=True)

    space(10)
    with st.expander("💬 Discussion"):
        st_disqus("bcithvacml")
