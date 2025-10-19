


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