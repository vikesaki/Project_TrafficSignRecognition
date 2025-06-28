import streamlit as st
import sidebar as sd
# Set page config
st.set_page_config(page_title="Traffic Sign Recognition", layout="wide", initial_sidebar_state="expanded")
page = sd.render_sidebar()

import pandas as pd
import numpy as np
import eda
import prediction 
import mainapp 



if page == "Home":
    mainapp.show()
elif page == "Classification Explanation":
    eda.show()
elif page == "Prediction":
    prediction.show()

