import streamlit as st
from rag_engine import query_assistant

st.set_page_config(page_title="Fashion RAG Assistant", layout="wide")
st.title("🛍️ Fashion Assistant (Myntra RAG Prototype)")

user_query = st.text_input("What are you looking for? (e.g. pastel kurtis under ₹1000)")

if user_query:
    with st.spinner("Searching the collection..."):
        response = query_assistant(user_query)
    st.markdown("### Recommendations")
    st.write(response)
