import streamlit as st
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, SequentialChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ========== File Processing ==========
def file_processing(file_path, job_text):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    curriculum_vitae = ''.join([page.page_content for page in data])

    splitter_cv = TokenTextSplitter(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks_cv = splitter_cv.split_text(curriculum_vitae)
    document_chunks = [Document(page_content=t) for t in chunks_cv]

    splitter_job = TokenTextSplitter(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=100
    )
    job_chunks = splitter_job.split_text(job_text)
    job_documents = [Document(page_content=t) for t in job_chunks]

    return document_chunks, job_documents


# ========== LLM Pipeline ==========
def llm_pipeline(file_path, job_text):
    document_chunks, job_chunks = file_processing(file_path, job_text)

    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    PROMPT_READ_RESUME = PromptTemplate(
        template=prompt_template,
        input_variables=["resume_text"]
    )
    COMPARE_PROMPT_RESUME = PromptTemplate(
        template=Comp_template,
        input_variables=["resume_text", "job_text"]
    )

    read_resume_chain = LLMChain(
        llm=llm,
        prompt=PROMPT_READ_RESUME,
        output_key="resume_summary"
    )
    compare_resume_chain = LLMChain(
        llm=llm,
        prompt=COMPARE_PROMPT_RESUME,
        output_key="comparison_result"
    )

    chain_prompt_resume = SequentialChain(
        chains=[read_resume_chain, compare_resume_chain],
        input_variables=["resume_text", "job_text"],
        output_variables=["resume_summary", "comparison_result"],
        verbose=True
    )

    embeddings = OpenAIEmbeddings()
    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    retriever = vectordb.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    resume_chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    results = []
    for i, resume_chunk in enumerate(document_chunks):
        for j, job_chunk in enumerate(job_chunks):
            output = chain_prompt_resume({
                "resume_text": resume_chunk.page_content,
                "job_text": job_chunk.page_content
            })
            results.append((
                f"Resume Chunk {i+1} vs Job Chunk {j+1}",
                output
            ))

    return resume_chat_chain, results


# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="Resume Intelligence", layout="wide")
    st.title("🧠 Resume Intelligence")
    st.subheader("Smart Resume vs. Job Match Analyzer — Powered by AI")

    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_text = st.text_area("📝 Paste Job Description", height=200)

    if st.button("🚀 Analyze Resume"):
        if uploaded_file is None:
            st.error("Please upload a resume file.")
            return
        if not job_text.strip():
            st.error("Please paste a job description.")
            return

        with st.spinner("Analyzing your resume against the job description..."):
            temp_pdf_path = "temp_resume.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                resume_chat_chain, results = llm_pipeline(temp_pdf_path, job_text)

                for title, output in results:
                    with st.expander(title):
                        st.write("**Summary:**")
                        st.write(output.get("resume_summary", ""))
                        st.write("**Comparison:**")
                        st.write(output.get("comparison_result", ""))

                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
