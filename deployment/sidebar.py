import streamlit as st

def render_sidebar():
    st.sidebar.title("Traffic Sign Recognition Project")
    st.sidebar.markdown("This dashboard lets you visualize and understand how traffic signs are grouped and classified using computer vision and deep learning.")
    st.sidebar.markdown("Upload your own images, see predictions in real-time, and learn how the model interprets different traffic sign types.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    selected = st.sidebar.radio("Go to", ["Home", "Classification Explanation", "Prediction"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("a project by *[vikesaki](github.com/vikesaki)*")
    
    return selected

