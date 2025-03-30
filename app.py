import streamlit as st
import pandas as pd
from together import Together
import math

# API Keys (Alternating)
API_KEYS = [
    "6fb2a9341f8f1a3505b9306cc21069016c787e625a2f42ddef9b772c5bfc214b",
    "db15e4757785a4c6a22e37360e60d04c2d9d08ac331d70bb093fd7977df5cc39",
]
api_index = 0  # Track which API key is being used

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1"  # Ensure DeepSeek is available
FILE_PATH = "4_updated_professor_data.csv"
BATCH_SIZE = 1000  # Adjust batch size based on dataset size and API limits

def get_client():
    """Get a Together AI client with the next API key."""
    global api_index
    api_key = API_KEYS[api_index % len(API_KEYS)]  # Rotate API keys
    api_index += 1
    return Together(api_key=api_key)

def together_request(prompt):
    """Send a request to Together AI using DeepSeek, rotating API keys."""
    client = get_client()  # Get the next client with a different API key
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: API request failed - {e}"

def process_batch(df_batch, user_query):
    """Send a batch of professor data to DeepSeek AI for ranking."""
    dataset_text = df_batch.to_csv(index=False)

    # Formulate the LLM prompt
    prompt = f"""
    You are an expert in recommending top professors for research collaboration.
    The user query is: "{user_query}".

    Here is a batch of professor data:
    {dataset_text}

    Your task:
    - Analyze the dataset.
    - Select the top 2 most relevant professors.
    - Provide a short reason (within 20 words) for each recommendation.

    Return the response in a structured format:
    - Name | Institution | Research Domain | h-index | i10-index | Citations | Short Reason
    """

    return together_request(prompt)

def batch_process_and_rank(user_query):
    """Divide dataset into batches, send requests, and aggregate responses."""
    df = pd.read_csv(FILE_PATH)
    num_batches = math.ceil(len(df) / BATCH_SIZE)
    all_responses = []

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        df_batch = df.iloc[start_idx:end_idx]
        batch_response = process_batch(df_batch, user_query)
        all_responses.append(batch_response)

    # Combine responses and pick top 5 overall
    aggregated_prompt = f"""
    Below are multiple professor recommendation lists from different dataset batches. Your task:
    - Merge all responses.
    - Rank the top 4 most relevant professors overall.
    - Keep explanations within 20 words.

    Responses:
    {"\n\n".join(all_responses)}

    Return the final list in this format:
    - Name | Institution| h-index | i10-index | Citations | Short Reason
    """
    
    return together_request(aggregated_prompt)

# Streamlit App
st.title("Professor Recommendation System")
st.write("Enter a research topic or area of interest to find top professors.")

user_query = st.text_input("Enter your research query:", "I want to collaborate on Generative AI and LLM Domains. List top professors.")

if st.button("Find Professors"):
    if user_query.strip():
        st.write("### AI-Generated Recommendations:")
        final_response = batch_process_and_rank(user_query)
        st.text_area("Professor Recommendations", final_response, height=300)
    else:
        st.warning("Please enter a valid query.")
