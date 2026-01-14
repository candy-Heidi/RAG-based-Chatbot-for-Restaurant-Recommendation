# RAG-based-Chatbot-for-Restaurant-Recommendation

This project is a Proof of Concept (PoC) for a **location-aware conversational search system**  
that combines **vector similarity search, geospatial filtering, and LLM-based reasoning**.

The goal is to demonstrate how user queries can be answered more accurately by  
**retrieving semantically similar places first and then using an LLM to generate responses based on retrieved context**.

---

## 📌 Project Overview

This system is designed to answer natural-language questions such as:

> “Find pet-friendly cafes near my current location with ratings above 3.”

Instead of relying on keyword-based search, the system:

- Filters candidates by **geographic proximity**
- Performs **semantic similarity search** using vector embeddings
- Uses an **LLM to generate responses grounded in retrieved place information**

---
Place Data (JSON)
↓
Text-based Place Description
↓
Embedding Generation
↓
Vector Storage (MongoDB)
↓
User Location + Query
↓
Geospatial Filtering
↓
Vector Similarity Search
↓
Context Construction
↓
LLM-based Answer Generation

---

## 🔹 Key Components

### 1. Place Embedding Generation

- Converts structured place data (name, category, rating, distance, features) into natural-language descriptions
- Generates vector embeddings using OpenAI embedding models
- Stores embeddings for later semantic search

This step allows the system to match user intent beyond exact keywords.

---

### 2. Location-based Candidate Filtering

- Uses latitude/longitude to calculate distances with the Haversine formula
- Narrows search scope to nearby places before vector similarity computation
- Improves both performance and relevance

---

### 3. Vector Similarity Search

- Computes cosine similarity between user query embeddings and place embeddings
- Retrieves the top-K most relevant nearby places
- Ensures responses are grounded in semantically relevant data

---

### 4. LLM-powered Conversational Response

- Constructs a structured context from retrieved places
- Uses LangChain with an OpenAI chat model
- Generates natural-language answers strictly based on retrieved context

This prevents hallucination and keeps responses data-driven.

---

## 🎯 What This Project Demonstrates

- Combining **geospatial filtering and vector similarity search**
- Using embeddings for semantic retrieval instead of keyword matching
- Grounding LLM responses in retrieved database context
- Applying RAG (Retrieval-Augmented Generation) patterns to real-world location data
- Designing scalable conversational search systems

---

## 📄 Related Publication

This code is based on research that has been published as an academic paper.  
The paper can be accessed at the link below.

- **Paper:** https://drive.google.com/drive/folders/1a5hwtDlY_KU2EV2XmC2gKNA-y4MxT-cS?usp=sharing
- **Language:** Korean

---

## ⚠ Notes

- This repository is intended for **concept demonstration purposes**
- API keys, database credentials, and production configurations are not included
- Additional optimization would be required for large-scale deployment

---

## 🧠 One-line Summary

> **A location-aware conversational search system that combines vector similarity search with LLM-based response generation.**


## 🏗 System Architecture

