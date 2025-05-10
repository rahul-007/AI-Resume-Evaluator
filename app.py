import os
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

# --- Gemini Model Initialization ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Prompt Templates ---
summary_resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
        Summarize the following resume by extracting key skills, job roles, achievements, and relevant experience:

        {resume}
        """
)

summary_jd_prompt = PromptTemplate(
    input_variables=["job_description"],
    template="""
        Summarize the following job description by extracting core responsibilities, qualifications, and required skills:

        {job_description}
        """
)

analysis_prompt = PromptTemplate(
    input_variables=["resume_summary", "jd_summary"],
    template="""
        You are an expert resume evaluator for an ATS system.
        Given the summarized resume:
        {resume_summary}

        And the summarized job description:
        {jd_summary}

        Perform the following analysis:
        1. What is the resume-to-job description match percentage?
        2. What are the missing skills from the resume based on the job description?
        3. Suggest improvements to the resume, specifically:
        - Add missing skills
        - Ensure each bullet point follows: what was done, why it was done, and what the impact was.
        - Any other improvements based on best resume practices (clarity, formatting, relevance, action verbs, etc).

        Respond in markdown format with headers for each section.
        """
)

# --- Chains ---
resume_summary_chain = LLMChain(llm=llm, prompt=summary_resume_prompt)
jd_summary_chain = LLMChain(llm=llm, prompt=summary_jd_prompt)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Evaluator (ATS)", layout="wide")
st.title("AI-Powered Resume Evaluator (ATS Optimizer)")

job_description = st.text_area("Paste the Job Description", height=150)
resume_file = st.file_uploader("Upload Your Resume (PDF Only)", type=["pdf"])

if st.button("Analyze Resume"):
    if not job_description or not resume_file:
        st.warning("Please provide both the job description and upload a resume.")
    else:
        # Extract text from PDF
        pdf_doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        resume_text = "\n".join(page.get_text() for page in pdf_doc)

        # Summarize resume and job description
        with st.spinner("Summarizing inputs with Gemini..."):
            resume_summary = resume_summary_chain.run(resume=resume_text)
            st.header("Resume Summary:")
            st.write(resume_summary)
            jd_summary = jd_summary_chain.run(job_description=job_description)
            st.header("JD Summary:")
            st.write(jd_summary)

        # Run final analysis
        with st.spinner("Analyzing summarized resume and job description..."):
            result = analysis_chain.run(resume_summary=resume_summary, jd_summary=jd_summary)

        # Display result
        st.markdown(result)
