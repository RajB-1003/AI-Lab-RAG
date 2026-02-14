from dataclasses import dataclass
from typing import List
import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    id: str
    text: str
    source: str

def load_chunks(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by policy headers (lines starting with #) to keep related content together
    sections = []
    current_section = []
    
    for line in text.split('\n'):
        # If line starts with # and we have accumulated content, save it as a chunk
        if line.strip().startswith('#') and current_section:
            sections.append('\n'.join(current_section).strip())
            current_section = [line]
        else:
            current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section).strip())
    
    # Filter out very short sections
    chunks = [s for s in sections if len(s.strip()) > 50]

    return [Document(f"chunk_{i}", c, filepath) for i, c in enumerate(chunks)]

class TfidfRAG:

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform([d.text for d in docs])

    def retrieve(self, question, k=3):

        q_vec = self.vec.transform([question])
        sims = cosine_similarity(q_vec, self.mat).flatten()
        idx = np.argsort(sims)[::-1][:k]

        return [self.docs[i] for i in idx]

def ask_llm(context, question):

    prompt = f"""
You are an HR assistant.

Answer ONLY from the policy context.
If not found, say: Not mentioned in company policy.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = ollama.generate(
        model='phi3',
        prompt=prompt
    )

    return response['response']

def main():
    docs = load_chunks("hr_policies.txt")

    rag = TfidfRAG(docs)

    print("\nOffline Employee RAG Ready (Phi-3 Local)")
    print("Type 'exit' to quit\n")

    while True:

        q = input("Ask HR: ")

        if q.lower() == "exit":
            break

        retrieved = rag.retrieve(q)

        context = "\n\n".join(d.text for d in retrieved)

        print("\n--- Answer ---\n")
        print(ask_llm(context, q))

if __name__ == "__main__":
    main()
