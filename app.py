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

def rank_professors(df, query):
    """Rank professors based on keyword matching and weighted scoring."""
    keywords = query.lower().split()
    df["match_score"] = df.apply(
        lambda row: sum(kw in (str(row["Research Interest 1"]) + str(row["Research Interest 2"])).lower() for kw in keywords),
        axis=1
    )
    
    # Weighted ranking formula
    df["final_score"] = df["match_score"] * 5 + df["h-index"] * 0.5 + df["i10-index"] * 0.3 + df["Citations"] * 0.2
    
    return df.sort_values(by="final_score", ascending=False).head(5)

def recommend_professors(user_query):
    """Process user queries and recommend professors."""
    df = pd.read_csv(FILE_PATH)
    df[["h-index", "i10-index", "Citations"]] = df[["h-index", "i10-index", "Citations"]].apply(pd.to_numeric, errors="coerce")
    
    ranked_professors = rank_professors(df, user_query)
    recommendations = []
    
    for _, row in ranked_professors.iterrows():
        prompt = f"""Provide a concise reason why this professor is recommended:
        - Name: {row["Name"]}
        - Institution: {row["College/Company"]}
        - Research Domain: {row["Research Interest 1"]}
        - h-index: {row["h-index"]}
        - i10-index: {row["i10-index"]}
        - Citations: {row["Citations"]}
        Keep the explanation within 20 words.
        """
        reason = together_request(prompt)
        recommendations.append({
            "Name": row["Name"],
            "Institution": row["College/Company"],
            "Research Domain": row["Research Interest 1"],
            "h-index": row["h-index"],
            "i10-index": row["i10-index"],
            "Citations": row["Citations"]
        })
    
    return pd.DataFrame(recommendations)

# Streamlit App
st.title("Professor Recommendation System")
st.write("Enter a research topic or area of interest to find top professors.")

user_query = st.text_input("Enter your research query:", "I want to collaborate on Generative AI and LLM Domains. List top professors.")

if st.button("Find Professors"):
    if user_query.strip():
        st.write("### Top Recommended Professors:")
        recommendations_df = recommend_professors(user_query)
        st.dataframe(recommendations_df)
    else:
        st.warning("Please enter a valid query.")
