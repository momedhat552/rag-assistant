import json
import math
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
INDEX_FILE = "index.json"


def load_documents(folder="docs"):
    docs = []
    for name in os.listdir(folder):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                docs.append((name, f.read()))
    return docs


def chunk_text(text, size=500, overlap=50):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        return 0.0
    return dot / (length_a * length_b)


def embed(texts):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def build_index(folder="docs", path=INDEX_FILE):
    chunks = []
    for name, text in load_documents(folder):
        for piece in chunk_text(text):
            chunks.append({"source": name, "text": piece})
    vectors = embed([c["text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return len(chunks)


def search(question, k=3, path=INDEX_FILE):
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    query_vector = embed([question])[0]
    scored = [(cosine_similarity(query_vector, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]

GEN_MODEL = "gemini-flash-lite-latest"


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for _, c in chunks)
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def answer(question, k=3):
    chunks = search(question, k)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=build_prompt(question, chunks),
    )
    sources = sorted({c["source"] for _, c in chunks})
    return response.text, sources

def main():
    if len(sys.argv) < 2:
        print('Usage: py rag.py index | ask "question" | "question"')
        sys.exit(1)
    if sys.argv[1] == "index":
        print(f"Indexed {build_index()} chunks")
    elif sys.argv[1] == "ask":
        if len(sys.argv) < 3:
            print('Usage: py rag.py ask "your question"')
            sys.exit(1)
        text, sources = answer(sys.argv[2])
        print(text)
        print("Sources:", ", ".join(sources))
    else:
        for score, chunk in search(sys.argv[1]):
            print(f"[{score:.2f}] {chunk['source']}")
            print(chunk["text"])
            print("---")


if __name__ == "__main__":
    main()