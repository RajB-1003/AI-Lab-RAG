from dataclasses import dataclass
from typing import List
import numpy as np
import ollama
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    id: str
    text: str


def load_chunks(fp: str) -> List[Document]:
    if not os.path.exists(fp):
        return []

    with open(fp, "r", encoding="utf-8") as f:
        text = f.read()

    parts = text.split("###")
    return [Document(f"s{i}", p.strip()) for i, p in enumerate(parts) if p.strip()]


class Retriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        if not docs:
            self.mat = None
            return

        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([d.text for d in docs])

    def search(self, q: str, k: int = 4):
        if self.mat is None:
            return []

        sims = cosine_similarity(self.vec.transform([q]), self.mat).flatten()
        idx = np.argsort(sims)[::-1][:k]
        return [self.docs[i].text for i in idx]


def ask_llm(context: str, q: str):
    prompt = f"""
You are a technical documentation assistant.
Answer ONLY using the provided documentation.
Provide code snippets and endpoints when relevant.

DOCUMENTATION:
{context}

QUESTION:
{q}
"""

    response = ollama.generate(
        model='phi3',
        prompt=prompt
    )

    return response['response']


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "api_docs.txt")

    docs = load_chunks(file_path)
    retriever = Retriever(docs)

    print("Technical Documentation Assistant (Offline)")
    print("Type 'exit' to quit")

    while True:
        q = input("\nDev Question: ")

        if q.lower() in ["exit", "quit"]:
            break

        ctx_list = retriever.search(q)

        if not ctx_list:
            print("No relevant documentation found.")
            continue

        ctx = "\n\n".join(ctx_list)

        print("\n--- Answer ---\n")
        print(ask_llm(ctx, q))


if __name__ == "__main__":
    main()
