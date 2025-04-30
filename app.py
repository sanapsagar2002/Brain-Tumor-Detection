import streamlit as st
import home
import about

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About"])

# Load the selected page
if page == "Home":
    home.run()
elif page == "About":
    about.run()
