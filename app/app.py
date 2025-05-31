import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.makedirs("../log", exist_ok=True)

import logging
logging.basicConfig(
    filename='../log/output.log',filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
for noisy_logger in [
    'google.generativeai',
    'google.api_core',
    'google.auth',
    'google.auth.transport.requests',
    'httpx',
    'urllib3',
    'tornado.access',
    'tornado.application',
    'tornado.general'
    ]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.ERROR)


import streamlit as st
from src.models.model import *
from src.Hypothesis_Analysis.complexity_score import *
st.title("Percolation Point Hypothesis Generator")

complexity = st.slider("Select hypothesis complexity", 1, 10, 2)
phenomenon = st.text_input("Enter phenomenon or topic from the literature")
file_media = st.file_uploader("Upload PDF of a literature", type="pdf")


if st.button("Generate Hypothesis"):
    logging.info(f"Complexity : {complexity}")
    logging.info(f"phenomenon: {phenomenon}")
    logging.info(f"File name : {file_media.name}")
    logging.info("Generate Hypothesis button clicked.")
    if not phenomenon:
        st.warning("Please enter a phenomenon or topic to guide hypothesis generation.")
        logging.warning("Phenomenon input is missing.")
    elif not file_media:
        st.warning("Please upload a PDF containing relevant scientific literature to form a hypothesis.")
        logging.warning("PDF upload is missing.")
    else:            
         with st.spinner("Generating hypothesis and calculating scores..."):
            hypothesis = generate_hypothesis(phenomenon, complexity, file_media)
            complexity_score = calculate_complexity_score(hypothesis) 
            llm_complexity_score = analyze_complexity(hypothesis)
            info_density = get_info_density(hypothesis)
            st.markdown(f"**Generated Hypothesis:** {hypothesis}")
            st.markdown(f"**Calculated Complexity Score:** {complexity_score} \t LLM Complexity Score : {llm_complexity_score} ")
            st.markdown(f"**Information Density:** {info_density} ")

            logging.info(f"Hypothesis generated: {hypothesis}")
            logging.info(f"Complexity Score: {complexity_score}, LLM Score: {llm_complexity_score}, Info Density: {info_density}")