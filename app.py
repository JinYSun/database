pip install streamlit_ketcher
import streamlit as st
import pandas as pd
from streamlit_ketcher import st_ketcher

# Page setup
st.set_page_config(page_title="Non-fullere acceptor", page_icon="🔋", layout="wide")
st.title("OSC Database")

# Connect to the Google Sheet
url = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=accept"
url1 = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=111"
df = pd.read_csv(url, dtype=str, encoding='utf-8')
df1 = pd.read_csv(url1, dtype=str, encoding='utf-8')
edited_df = st.data_editor(df1, num_rows="dynamic")
st.download_button(
    "⬇️ Download edited files as .csv", edited_df.to_csv(), "edited_df.csv", use_container_width=True
)

molecule = st.text_input("Molecule", DEFAULT_MOL)
smile_code = st_ketcher(molecule)
st.markdown(f"Smile code: ``{smile_code}``")
