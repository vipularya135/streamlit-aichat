import streamlit as st
import pandas as pd
from together import Together

# Initialize Together AI Client
API_KEY = "6fb2a9341f8f1a3505b9306cc21069016c787e625a2f42ddef9b772c5bfc214b"  # Replace with your actual API key
client = Together(api_key=API_KEY)

# Constants
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
FILE_PATH = "4_updated_professor_data.csv"

def together_request(prompt):
    """Send a request to Together AI using the specified model."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: API request failed - {e}"

def generate_llm_response(user_query):
    """Send dataset and user query to LLM for ranking and recommendations."""
    df = pd.read_csv(FILE_PATH)
    df[["h-index", "i10-index", "Citations"]] = df[["h-index", "i10-index", "Citations"]].apply(pd.to_numeric, errors="coerce")

    # Convert dataset into a structured text format
    dataset_text = df.to_csv(index=False)

    # Formulate the LLM prompt
    prompt = f"""
    You are an expert in recommending top professors for research collaboration.
    Here is a dataset of professors with details such as Name, Institution, Research Domain, h-index, i10-index, and Citations.
    The user is looking for collaboration based on this query: "{user_query}".

    Your task:
    - Analyze the dataset.
    - Rank the top 5 most relevant professors.
    - Provide a short reason (within 20 words) for each recommendation.

    Here is the dataset:
    {dataset_text}

    Return the response in a structured format with:
    - Name | Institution | Research Domain | h-index | i10-index | Citations | Short Reason
    """

    return together_request(prompt)

# Streamlit App
st.title("Professor Recommendation System")
st.write("Enter a research topic or area of interest to find top professors.")

user_query = st.text_input("Enter your research query:", "I want to collaborate on Generative AI and LLM Domains. List top professors.")

if st.button("Find Professors"):
    if user_query.strip():
        st.write("### AI-Generated Recommendations:")
        llm_response = generate_llm_response(user_query)
        st.text_area("Professor Recommendations", llm_response, height=300)
    else:
        st.warning("Please enter a valid query.")
