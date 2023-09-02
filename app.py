
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Python Talks Search Engine", page_icon="🐍", layout="wide")
st.title("Python Talks Search Engine")

# Connect to the Google Sheet
url = f"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=accept"
df = pd.read_csv(url, dtype=str)
st.write(df)
