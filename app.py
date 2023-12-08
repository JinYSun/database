# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import rdkit
import streamlit_ketcher
from streamlit_ketcher import st_ketcher
import abcBERT
import RF


# Page setup
st.set_page_config(page_title="DeepAcceptor", page_icon="🔋", layout="wide")
st.title("🔋DeepAcceptor")

# Connect to the Google Sheet
url1 = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=accept"
url = r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw/gviz/tq?tqx=out:csv&sheet=111"
df1 = pd.read_csv(url1, dtype=str, encoding='utf-8')
col1, col2 = st.columns(2)
with col1:
	st.subheader("🔍**Search papers or molecules**")
	text_search = st.text_input(label="_", value="",label_visibility="hidden" )
	m1 = df1["name"].str.contains(text_search)
	m2 = df1["reference"].str.contains(text_search)
	df_search = df1[m1 | m2]
with col2:

	st.link_button("📝**DATABASE**",  r"https://docs.google.com/spreadsheets/d/1YOEIg0nMTSPkAOr8wkqxQRLuUhys3-J0I-KPEpmzPLw")
	st.markdown('👆If you want to update the database, click the button.')
if text_search:
    st.write(df_search)
    st.download_button( "⬇️ Download edited files as .csv", df_search.to_csv(), "df_search.csv", use_container_width=True)
edited_df = st.data_editor(df1, num_rows="dynamic")
edited_df.to_csv(url)
st.download_button(
    "⬇️ Download edited files as .csv", edited_df.to_csv(), "edited_df.csv", use_container_width=True
)
st.header("📋**Input the SMILES of Molecule**")
col3, col4= st.columns(2)

with col3:
	
	molecule = st.text_input(label="*",label_visibility="hidden")
with col4:
	st.markdown('👇An example of Y6.')
	if st.button("🙋‍♂️**Example**"):
    		molecule = 'O=C(C(C=C(F)C(F)=C1)=C1C/2=C(C#N)/C#N)C2=C/C3=C(CCCCCCCCCCC)C(S4)=C(S3)C5=C4C6=C(N5CC(CC)CCCC)C7=C(C(SC8=C9SC(/C=C%10C(C(C=C(F)C(F)=C%11)=C%11C\%10=C(C#N)C#N)=O)=C8CCCCCCCCCCC)=C9N7CC(CC)CCCC)C%12=NSN=C6%12'

smile_code = st_ketcher(molecule)
st.subheader(f"✨**Smiles code**: {smile_code}")
mol = rdkit.Chem.MolFromSmiles(smile_code)
if  mol is None:
		st.subheader('**❗The SMILES is ERROR❗**')
else:
	try :
		P = RF.main( str(smile_code ) )
		st.subheader(f"⚡**PCE predicted by RF**: {P}")
	except:

		st.subheader(f"⚡**PCE predicted by RF**: [Running]")
	try:
		pce = abcBERT.main( str(smile_code ) )
		st.subheader(f"⚡**PCE predicted by abcBERT**: {pce}")
	except:
		st.subheader(f"⚡**PCE predicted by abcBERT**:  [Running]")
