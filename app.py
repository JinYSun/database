
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Non-fullere acceptor", page_icon="🔋", layout="wide")
st.title("OSC Database")

# Connect to the Google Sheet
url = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=accept"
url1 = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=111"
df = pd.read_csv(url, dtype=str, encoding='utf-8')
df1 = pd.read_csv(url1, dtype=str, encoding='utf-8')
edited_df = st.data_editor(df)
edited_df = st.data_editor(df, num_rows="dynamic")
st.data_editor(df, key="data_editor") 
st.write("Here's the session state:")
st.write(st.session_state["data_editor"])
st.write(df,df1)

