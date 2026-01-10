#!/usr/bin/env python3
"""
RAG Vector Store Creator for Insurellm (OpenRouter Version)
Converted from day2.ipynb to standalone Python script
Creates vector embeddings and visualizations of the knowledge base
"""

import glob
import os

import numpy as np
import plotly.graph_objects as go
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sklearn.manifold import TSNE


def main():
    # Configuration
    MODEL = "deepseek/deepseek-chat"  # Cost-effective model via OpenRouter
    db_name = "vector_db_openrouter"

    # Load environment variables
    load_dotenv(override=True)

    # Setup OpenRouter client
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        print(f"OpenRouter API Key exists and begins {openrouter_api_key[:8]}")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    else:
        print("OpenRouter API Key not set")
        return

    print(f"Using model: {MODEL}")

    # PART A: Analyze document size
    print("\n=== PART A: Document Analysis ===")

    knowledge_base_path = "knowledge-base/**/*.md"
    files = glob.glob(knowledge_base_path, recursive=True)
    print(f"Found {len(files)} files in the knowledge base")

    entire_knowledge_base = ""
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            entire_knowledge_base += f.read()
            entire_knowledge_base += "\n\n"

    print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")

    # Token count analysis
    encoding = tiktoken.get_encoding("cl100k_base")  # DeepSeek uses similar encoding
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    print(f"Total tokens for {MODEL}: {token_count:,}")

    # Load documents using LangChain
    print("\n=== Loading Documents ===")
    folders = glob.glob("knowledge-base/*")

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    print(f"Loaded {len(documents)} documents")

    # Chunk the documents
    print("\n=== Chunking Documents ===")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Divided into {len(chunks)} chunks")
    print(f"First chunk preview: {str(chunks[0])[:200]}...")

    # PART B: Create vector store
    print("\n=== PART B: Creating Vector Store ===")

    # Use HuggingFace embeddings for cost efficiency
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clean up existing database
    if os.path.exists(db_name):
        try:
            Chroma(
                persist_directory=db_name, embedding_function=embeddings
            ).delete_collection()
        except:
            pass

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_name
    )
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

    # Investigate vectors
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )

    # PART C: Visualize vectors
    print("\n=== PART C: Creating Visualizations ===")

    # Get all vectors and metadata
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents_text = result["documents"]
    metadatas = result["metadatas"]
    doc_types = [metadata["doc_type"] for metadata in metadatas]

    # Color mapping
    color_map = {
        "products": "blue",
        "employees": "green",
        "contracts": "red",
        "company": "orange",
    }
    colors = [color_map.get(t, "gray") for t in doc_types]

    # 2D Visualization
    print("Creating 2D visualization...")
    tsne_2d = TSNE(n_components=2, random_state=42)
    reduced_vectors_2d = tsne_2d.fit_transform(vectors)

    fig_2d = go.Figure(
        data=[
            go.Scatter(
                x=reduced_vectors_2d[:, 0],
                y=reduced_vectors_2d[:, 1],
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.8),
                text=[
                    f"Type: {t}<br>Text: {d[:100]}..."
                    for t, d in zip(doc_types, documents_text)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig_2d.update_layout(
        title="2D Chroma Vector Store Visualization (OpenRouter Version)",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40),
    )

    # Save 2D plot
    fig_2d.write_html("vector_visualization_2d_openrouter.html")
    print("2D visualization saved as 'vector_visualization_2d_openrouter.html'")

    # 3D Visualization
    print("Creating 3D visualization...")
    tsne_3d = TSNE(n_components=3, random_state=42)
    reduced_vectors_3d = tsne_3d.fit_transform(vectors)

    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=reduced_vectors_3d[:, 0],
                y=reduced_vectors_3d[:, 1],
                z=reduced_vectors_3d[:, 2],
                mode="markers",
                marker=dict(size=5, color=colors, opacity=0.8),
                text=[
                    f"Type: {t}<br>Text: {d[:100]}..."
                    for t, d in zip(doc_types, documents_text)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig_3d.update_layout(
        title="3D Chroma Vector Store Visualization (OpenRouter Version)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        width=900,
        height=700,
        margin=dict(r=10, b=10, l=10, t=40),
    )

    # Save 3D plot
    fig_3d.write_html("vector_visualization_3d_openrouter.html")
    print("3D visualization saved as 'vector_visualization_3d_openrouter.html'")

    # Show plots if running interactively
    try:
        fig_2d.show()
        fig_3d.show()
    except:
        print("Note: Interactive plots require a browser environment")

    print("\n=== Summary ===")
    print(f"✓ Processed {len(files)} files")
    print(f"✓ Created {len(chunks)} chunks")
    print(f"✓ Generated {count:,} vectors with {dimensions:,} dimensions")
    print(f"✓ Saved vector database to '{db_name}'")
    print("✓ Created 2D and 3D visualizations")
    print(f"✓ Using {MODEL} via OpenRouter for cost efficiency")


if __name__ == "__main__":
    main()
