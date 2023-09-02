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
edited_df = st.data_editor(df1, num_rows="dynamic")
st.download_button(
    "⬇️ Download annotations as .csv", annotated.to_csv(), "annotated.csv", use_container_width=True
)
iframe(
        src="https://docs.google.com/spreadsheets/d/1Z0zd-5dF_HfqUaDDq4BWAOnsdlGCjkbTNwDZMBQ1dOY/edit#gid=0",
        height=600,
    )
import streamlit as st
from streamlit_ketcher import st_ketcher

molecule = st.text_input("Molecule", DEFAULT_MOL)
smile_code = st_ketcher(molecule)
st.markdown(f"Smile code: ``{smile_code}``")
