import pandas as pd
import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("tickets.csv")

df = df[df["status"] == "Closed"].copy()

for col in ["title", "description", "category", "resolution"]:
    df[col] = df[col].fillna("")

df["search_text"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["category"]
)

texts = df["search_text"].tolist()

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(texts)

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx = np.argsort(sims)[::-1][:k]
    return df.iloc[idx]

def ask_llm(context, question):
    prompt = f"""
Use ONLY the past ticket resolutions to write a professional reply.

PAST RESOLUTIONS:
{context}

NEW ISSUE:
{question}
"""
    response = ollama.generate(
        model='phi3',
        prompt=prompt
    )
    return response['response']

def main():
    print("Customer Support Ticket Autocomplete (Offline)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nTicket: ")

        if query.lower() == "exit":
            break

        results = retrieve(query)

        if results.empty:
            print("No similar tickets found.")
            continue

        context = "\n".join([row["resolution"] for _, row in results.iterrows()])

        print("\n--- Suggested Reply ---\n")
        print(ask_llm(context, query))

if __name__ == "__main__":
    main()
