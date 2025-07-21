import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer("intfloat/e5-base-v2")
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection("products")

def query_assistant(user_query: str) -> str:
    query_embedding = model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a fashion assistant. A customer is looking for fashion items.

Context:
{context}

User Query:
{user_query}

Recommend suitable items from the above and explain why.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
