# GATE-Rec 
(Gated Adaptive Text-Enhanced Recommender)
### Deep Recommender Framework for Cold-Start Mitigation and Cross-/Up-Selling

GATE-Rec is a hybrid deep recommendation framework developed as part of my MSc Computer Science dissertation. It combines behavioural information from user interaction sequences with product text information to address the **item cold-start problem** while supporting **cross-selling and up-selling** applications.

## System Architecture

#![GATE-Rec System Architecture](docs/GATE-Rec_System_Architecture.png)

## Problem

Traditional recommendation systems depend heavily on historical user-item interactions. This creates a **cold-start problem** for newly introduced products with little or no interaction history.

At the same time, recommendation systems can support business applications such as:

- **Cross-selling** – recommending complementary or related products
- **Up-selling** – recommending higher-rated alternatives within the same category

GATE-Rec addresses these requirements through a unified hybrid embedding framework.

## Objectives

1. Develop a hybrid recommendation framework combining **GRU4Rec behavioural embeddings** with **SentenceTransformer textual embeddings** using a gated fusion mechanism.
2. Evaluate the proposed framework against multiple baseline models under a cold-start recommendation setting across three product categories.
3. Analyse the effectiveness of the hybrid embeddings for **cold-start recommendation, cross-selling, and up-selling** through retrieval performance and embedding visualisations.

## Approach

The framework follows a two-phase pipeline:

**Phase 1 – Behavioural Embedding Learning**
- User interaction sequences are processed chronologically.
- **GRU4Rec** learns sequential item representations from interaction history.
- Behavioural item embeddings are extracted from the trained model.

**Phase 2 – Hybrid Embedding Construction**
- Product metadata is converted into text representations using **SentenceTransformer (all-MiniLM-L6-v2)**.
- Behavioural and textual representations are projected into a common space.
- A learned **gating mechanism** dynamically balances behavioural and textual information.
- For cold items without behavioural history, the model relies on the available textual signal.
- The resulting 128-dimensional hybrid embeddings are used for recommendation and similarity-based retrieval.

## Dataset

Experiments were conducted using the **Amazon Reviews 2023** dataset across:

- Electronics
- Office Products
- Beauty & Personal Care

Both review interactions and product metadata were used to construct behavioural and content-based representations.

## Evaluation

The framework was evaluated using:

- **Hit@10**
- **NDCG@10**
- **Cold Efficiency**
- **Warm–Cold Gap**
- **Harmonic Mean**
- PCA and t-SNE embedding visualisations

Multiple baselines were also evaluated, including Random, Most Popular, BPR-MF, GRU4Rec, Text-Only retrieval, and SASRec.

## Selected Results

| Category | Warm Hit@10 | Cold Hit@10 | Cold Retention |
|---|---:|---:|---:|
| Electronics | 0.8962 | 0.8900 | 99.3% |
| Office Products | 0.9094 | 0.8923 | 98.1% |
| Beauty & Personal Care | 0.8772 | 0.8462 | 96.5% |

The hybrid embeddings also supported similarity-based **cross-selling and up-selling retrieval**, with high catalogue coverage across the evaluated categories.

## Technologies

- Python
- PyTorch
- GRU4Rec
- SentenceTransformers
- Hugging Face Transformers
- NumPy
- Pandas
- Scikit-learn
- Google Colab

## Repository Structure

```text
GATE-Rec/
│
├── notebooks/
│   ├── Electronics/
│   ├── Office_Products/
│   └── Beauty_and_Personal_Care/
│
├── results/
│   ├── visualizations/
│   │   ├── PCA/
│   │   └── t-SNE/
│   ├── cross_up_selling/
│   └── baseline_comparison/
│       ├── recommendation_performance/
│       └── cold_start_evaluation/
│
└── README.md
