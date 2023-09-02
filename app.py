
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Python Talks Search Engine", page_icon="🐍", layout="wide")
st.title("Python Talks Search Engine")

# Connect to the Google Sheet
url = f"https://docs.google.com/spreadsheets/d/1x1z3l-Hxcx5mwkTYM_AVaqzHt7nBfOOWQphnaoTAkLY/edit?usp=sharing"
df = pd.read_csv(url, dtype=str)

# Show the dataframe (we'll delete this later)
st.write(df)
