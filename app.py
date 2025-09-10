from main import run_pipeline
import os
import streamlit as st

SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("PANALIZER")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = run_pipeline(save_path)  # returns a dict

    st.subheader("Summary:")
    st.write(result["summary"])

    st.subheader("Key Clauses:")
    for clause in result["clauses"]:
        st.write(f"- {clause}")
