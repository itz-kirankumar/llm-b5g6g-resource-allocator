import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings



#dataset splitting


loader = PyPDFLoader("/content/ai pc.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

text = RecursiveCharacterTextSplitter().split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
encode_kwargs={"normalize_embeddings": True})

#model training

from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(text, embeddings)
#hyperparameter tuing
#hybrid search
from langchain.retrievers import BM25Retriever, EnsembleRetriever
     

keyword_retriever = BM25Retriever.from_documents(text)
     

keyword_retriever.k =  3
     

ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore,keyword_retriever],weights=[0.3, 0.7]

model_name = "HuggingFaceH4/zephyr-7b-beta"
def load_quantized_model(model_name: str):
    """
    model_name: Name or path of the model to be loaded.
    return: Loaded quantized model.
    """
# initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    model_name: Name or path of the model for tokenizer initialization.
    return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer
     

tokenizer = initialize_tokenizer(model_name)

from sentence_transformers import CrossEncoder
     

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = cross_encoder.predict(pairs)
reranked_document_cross_encoder = sorted(scored_docs, reverse=True)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=knowledge_base.as_retriever())
vectorstore.save_local("vectorstore.db")

retriever=vectorstore.as_retriever()

# Set your Perplexity API key
API_KEY = "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"
API_URL = "https://api.perplexity.ai/chat/completions"

# Function to query Perplexity API
def query_perplexity(user_ids, qos_requirements, available_bandwidth, available_power):
    prompt = f"""
    You are an expert in wireless communication systems. Allocate bandwidth and power resources to users in a B5G network based on their QoS requirements.

    Input:
    - User IDs: {user_ids}
    - QoS Requirements: {qos_requirements}
    - Available Bandwidth (MHz): {available_bandwidth}
    - Available Power (Watts): {available_power}

    Output:
    Provide a resource allocation plan with bandwidth and power assigned to each user.
    """

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

# Function to parse API response into structured data
def parse_allocation_plan(response_text):
    try:
        lines = response_text.split("\n")
        data = []
        
        for line in lines:
            if line.startswith("User_"):
                parts = line.split(":")
                user_info = parts[0].strip()
                allocations = parts[1].split(",")
                bandwidth = int(allocations[0].split()[0])
                power = int(allocations[1].split()[0])
                qos = user_info.split("(")[-1].replace("QoS)", "").strip()
                data.append({"User ID": user_info.split()[0], 
                             "Bandwidth (MHz)": bandwidth, 
                             "Power (Watts)": power, 
                             "QoS Requirement": qos})

        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to parse API response: {e}")
        return None

# Streamlit app UI
st.title("Real-Time Resource Allocation for B5G Networks")

# Sidebar inputs
st.sidebar.header("Input Parameters")
num_users = st.sidebar.slider("Number of Users", min_value=1, max_value=10, value=5)
available_bandwidth = st.sidebar.number_input("Available Bandwidth (MHz)", min_value=10, max_value=1000, value=100)
available_power = st.sidebar.number_input("Available Power (Watts)", min_value=10, max_value=500, value=50)
qos_requirements = [st.sidebar.selectbox(f"QoS for User_{i+1}", ["High", "Medium", "Low"], key=f"qos_user_{i+1}") for i in range(num_users)]
user_ids = [f"User_{i+1}" for i in range(num_users)]

# Main content layout
if st.button("Allocate Resources"):
    allocation_plan_text = query_perplexity(user_ids, qos_requirements, available_bandwidth, available_power)

    if allocation_plan_text:
        # Parse API response into DataFrame
        allocation_df = parse_allocation_plan(allocation_plan_text)

        if allocation_df is not None:
            # Dynamically calculate summary statistics
            total_bandwidth_used = allocation_df["Bandwidth (MHz)"].sum()
            total_power_used = allocation_df["Power (Watts)"].sum()
            remaining_bandwidth = available_bandwidth - total_bandwidth_used
            remaining_power = available_power - total_power_used

            # Display Summary Section
            st.subheader("Summary")
            st.write(f"**Total Bandwidth Used:** {total_bandwidth_used} MHz")
            st.write(f"**Total Power Used:** {total_power_used} Watts")
            st.write(f"**Remaining Bandwidth:** {remaining_bandwidth} MHz")
            st.write(f"**Remaining Power:** {remaining_power} Watts")

            # Display Allocation Table
            st.subheader("Resource Allocation Table")
            st.dataframe(allocation_df)

            # Visualization: Horizontal Stacked Bar Chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(allocation_df["User ID"], allocation_df["Bandwidth (MHz)"], label="Bandwidth (MHz)", color="skyblue")
            ax.barh(allocation_df["User ID"], allocation_df["Power (Watts)"], left=allocation_df["Bandwidth (MHz)"], label="Power (Watts)", color="orange")
            
            ax.set_xlabel("Resources Allocated")
            ax.set_title("Resource Allocation Visualization")
            ax.legend()
            
            st.pyplot(fig)
)