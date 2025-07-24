from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration

with open("./data/paragraphs.txt", "r", encoding="utf-8") as file:
    paragraphs = file.read().split("\n\n")


documents = []
for para in paragraphs:
    documents.extend(sent_tokenize(para))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

# Normalize embeddings for cosine similariies
doc_embeddings = doc_embeddings/np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# Build FAISS index with inner product
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings.astype('float32'))

# Question > embeddings > normalize embeddings
question = "Where was lumo living?"
q_embeddings = embedder.encode([question])
q_embeddings = q_embeddings/np.linalg.norm(q_embeddings, axis=1, keepdims=True)

# D = dot products (Similarity scores), I = indexes k = 5 (top 5 results)
D, I = index.search(q_embeddings.astype('float32'), k=5)

print(f"\n\nTop retrieved sentences and similarity scores:")
for score, idx in zip(D[0], I[0]):
    print(f"{score:.4f}: {documents[idx]}")

# Join top 5 sentences into one for context
context = " ".join(documents[idx] for idx in I[0])
print(f"\nRetrieved context: {context}")

# T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
generator = T5ForConditionalGeneration.from_pretrained('t5-small')

# prompt for T5 model
input_text = f"question: {question} context: {context}"

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding="max_length"
)

print(f"\nInput token length: {inputs.input_ids.size(1)}")

# Generate answer
output_ids = generator.generate(
    inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_length = 150,
    min_length = 10,
    num_beams = 8,
    no_repeat_ngram_size = 1,
    early_stopping = True,
    length_penalty = 1.5
)

# Final answer
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\n\nAnswer: {answer}\n\n")
