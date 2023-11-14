# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import rdkit
import streamlit_ketcher
from streamlit_ketcher import st_ketcher
import abcBERT
import RF
from streamlit_gsheets import GSheetsConnection

# Page setup
st.set_page_config(page_title="DeepAcceptor", page_icon="🔋", layout="wide")
st.title("🔋DeepAcceptor")

# Connect to the Google Sheet
url1 = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=accept"
url = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=111"
df1 = pd.read_csv(url1, dtype=str, encoding='utf-8')

text_search = st.text_input("🔍Search papers or molecules", value="")
m1 = df1["name"].str.contains(text_search)
m2 = df1["reference"].str.contains(text_search)
df_search = df1[m1 | m2]
if text_search:
    st.write(df_search)
    st.download_button( "⬇️ Download edited files as .csv", df_search.to_csv(), "df_search.csv", use_container_width=True)
edited_df = st.data_editor(df1, num_rows="dynamic")
edited_df.to_csv(url)
st.download_button(
    "⬇️ Download edited files as .csv", edited_df.to_csv(), "edited_df.csv", use_container_width=True
)

molecule = st.text_input("📋Molecule")
smile_code = st_ketcher(molecule)
st.markdown(f"✨Smiles code: {smile_code}")
P = RF.main( str(smile_code ) )
st.markdown(f"⚡PCE predicted by RF: {P}")

try:
    pce = abcBERT.main( str(smile_code ) )
    st.markdown(f"⚡PCE predicted by abcBERT: {pce}")
except:
    st.markdown(f"⚡PCE predicted by abcBERT:  Running")

