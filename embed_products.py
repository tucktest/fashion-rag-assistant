import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load dataset
df = pd.read_csv("products.csv")

# Preprocess
df["text"] = (
    "Title: " + df["productDisplayName"].fillna("") +
    "\nBrand: " + df["brandName"].fillna("") +
    "\nGender: " + df["gender"].fillna("") +
    "\nCategory: " + df["masterCategory"].fillna("") +
    "\nSubCategory: " + df["subCategory"].fillna("") +
    "\nDescription: " + df["productDetails"].fillna("")
)

# Init embedding
model = SentenceTransformer("intfloat/e5-base-v2")

# Init Chroma
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection("products")

# Add embeddings
for i, row in df.iterrows():
    embedding = model.encode(row["text"])
    collection.add(documents=[row["text"]], embeddings=[embedding.tolist()], ids=[str(i)])

print("✅ Embedding complete.")
