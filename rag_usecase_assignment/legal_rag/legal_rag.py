import numpy as np
import ollama
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "contract.txt")

if not os.path.exists(file_path):
    print("contract.txt not found")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

clauses = text.split("\n\n")
clauses = [c.strip() for c in clauses if len(c.strip()) > 10]

if not clauses:
    print("No clauses found")
    exit(1)

vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
mat = vec.fit_transform(clauses)

def search(q, k=3):
    sims = cosine_similarity(vec.transform([q]), mat).flatten()
    idx = np.argsort(sims)[::-1][:k]
    return [clauses[i] for i in idx]

def ask(context, q):
    prompt = f"""
Answer ONLY using the legal clauses below.

{context}

Question: {q}
"""
    response = ollama.generate(
        model='phi3',
        prompt=prompt
    )
    return response['response']

def main():
    print("Legal Clause Finder RAG Loaded.")
    print("Type 'exit' to quit.")

    while True:
        q = input("\nLegal Query: ")
        if q.lower() in ["exit", "quit"]:
            break

        results = search(q)

        if not results:
            print("No relevant clauses found.")
            continue

        ctx = "\n\n".join(results)
        print(ask(ctx, q))

if __name__ == "__main__":
    main()
