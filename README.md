
<h1 align="center">🧠 Resume-Job Matcher</h1>
<p align="center">Smart Resume vs. Job Match Analyzer — Powered by AI</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/LangChain-Enabled-green?logo=chainlink">
  <img src="https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-black?logo=openai">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

> 💼 AI-powered system that reads a candidate's resume, compares it to a job description, and provides detailed feedback, match scoring — using **LangChain**, **OpenAI**, and **ChromaDB**.

---

## 🚀 Features

- ✅ Extract resume content from PDF
- ✅ Chunk resume and job description using token-aware splitting
- ✅ Generate smart resume summaries
- ✅ Compare resume to job description via custom prompt chains
- ✅ Store and retrieve resume embeddings with ChromaDB
- ✅ Chat with the resume using retrieval-based memory
- ✅ Modular, extensible, and OpenAI-compatible

---

## 📁 Project Structure

```text
.
├── .DS_Store
├── .env
├── .gitignore
├── Data
│   └── Chike's CV.pdf
├── README.md
├── Resume_Matcher_Project.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── app.py
├── db
│   
│      
│     
│    
│     
│   
├── requirements.txt
├── research
│   └── experiment.ipynb
├── setup.py
├── src
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
├── structure.txt
└── temp_resume.pdf
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/obinnachike/match_resume
cd match_resume
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

💡 **Note:** Never commit your `.env` file to Git. Add it to `.gitignore`.

---

## How It Works

###  1. Upload Resume (PDF)
- Resume is parsed with `PyPDFLoader`
- Chunked using `TokenTextSplitter` for LLM processing

###  2. Paste Job Description
- The job description is also token-split and prepared for comparison

###  3. LangChain LLM Pipeline
- Summarize resume with `LLMChain`
- Compare resume vs job using `SequentialChain`
- Store resume embeddings in `ChromaDB`
- Enable Q&A with `ConversationalRetrievalChain`

###  4. Get Comparison Results
- Per-chunk analysis
- Match findings
- Optional Q&A interface

---


---

## ✍️ Prompt Customization

Edit the file `src/prompt.py`:

```python
prompt_template = """
You are an expert career assistant

------------
Curriculum vitae:
{resume_text}
------------

Read the content of Curriculum vitae and be sure to understand the content, 
while applying you roles. Respond 'YES' when you have read and understood the Curriculum vitae

"""

Comp_template = ("""
You have analyzed and understood the content of curriculum vitae. As an expert career assistant
Your roles are to compare Curriculum vitae with Job description and return the following:
        1. A **Fit Score** (0-100%) of how well this resume matches the job.
        2. Key strengths (resume areas that align well).
        3. Specific recommendations to improve the resume to better fit the job.
        Format neatly in Markdown. 
------------
Job description:
{job_text}
------------
"""
)

job_text = """
We are hiring a research associate with strong background in pharmacology and data interpretation.
Knowledge of clinical trial analysis and regulatory documentation is a plus.
"""
```

---

## Example Usage

```python
from main import llm_pipeline

resume_path = " "
job_description = """ """

chatbot, results = llm_pipeline(resume_path, job_description)

for section, result in results:
    print(f"{section}:\n{result}\n")
```

---


## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Shields.io](https://shields.io) for badges
