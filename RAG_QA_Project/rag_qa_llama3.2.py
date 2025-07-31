from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from chromadb import PersistentClient
import numpy as np, string
import requests
import json

class RAG_QA:
    def __init__(self):
        self.documents = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')


    def load_documents(self):
        with open("./data/paragraphs.txt", "r", encoding="utf-8") as f:
            paragraphs = f.read().split("\n\n")

        # for para in paragraphs:
            # self.documents.extend(sent_tokenize(para))

        for para in paragraphs:
            sents = sent_tokenize(para)
            # self.documents.extend(sents)
            chunk = ""
            for sent in sents:
                if len(chunk.split()) + len(sent.split()) <= 180:
                    chunk += " " + sent
                else:
                    self.documents.append(chunk.strip())
                    chunk = sent
            if chunk:
                self.documents.append(chunk.strip())

    def embed_documents(self):
        embeddings = self.embedder.encode(self.documents)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def create_collection(self, embeddings):
        client = PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("docs")
        collection.add(documents=self.documents, embeddings=embeddings, ids=[str(i) for i in range(len(self.documents))])
        return collection

    def retrieve_context(self, question, collection, top_k=8):
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        results = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)
        return " ".join(results["documents"][0]), results

    def generate_answer(self, question, context):
        # input_text = f"question: {question} context: {context}"
        # input_text = f"""Answer using only the provided context, in under 10 words. The answer should be meaningful and help understand the situation.
        
        # Context: {context} 
        
        # Question: {question} 
        
        # Answer:"""

        input_text = f"""Question: {question}

            Answer using only the provided context. Keep it under 10 words.
            Your answer should:
                •	Be meaningful, not generic.
                •	Reflect key details (species, actions, consequences).
                •	Prioritize expected or implied elements over surface-level summary.
                •	Reflect key values or turning points (e.g., growth, responsibility, ethics, etc.).
                •	Prioritize content that aligns with the central themes or changes

            Context: {context}

            Answer:"""
       
        # print(f"*** \n{input_text}\n ***")


        return self.get_response_from_llama(input_text)
    
       
    def bleu_scores(self, answer, references):
        ans_tokens = [w for w in word_tokenize(answer.lower()) if w not in string.punctuation]
        ref_tokens = [[w for w in word_tokenize(ref.lower()) if w not in string.punctuation] for ref in references]
        smoothie = SmoothingFunction().method4
        scores = [
            sentence_bleu(ref_tokens, ans_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        ]
        print(f"\nBLEU Scores:")
        for i, score in enumerate(scores, 1):
            print(f"BLEU-{i}: {score:.2f}")


    def get_response_from_llama(self, query_prompt: str, model: str="llama3.2"):
        response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": query_prompt,
            "stream":   False
         }
         )
    
        if response.status_code == 200:
            return response.json()["response"].strip()



if __name__ == "__main__":


# Questions and multiple reference answers for each
   
   # Load questions and references from file
    with open("./data/qa_pairs.json", "r", encoding="utf-8") as f:
        all_qa_pairs = json.load(f)

    question_indices_to_run = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    # question_indices_to_run = [9]  

    if question_indices_to_run:
        qa_pairs = [all_qa_pairs[i] for i in question_indices_to_run if i < len(all_qa_pairs)]
    else:
        qa_pairs = all_qa_pairs



    rag = RAG_QA()
    rag.load_documents()
    doc_embeds = rag.embed_documents()
    collection = rag.create_collection(doc_embeds)

    for qa in qa_pairs:
        print(f"\n{'='*80}")
        print(f"Question: {qa['question']}")
        context, results = rag.retrieve_context(qa["question"], collection)
        
        # print("\nRetrieved Sentences:")
        # for doc, dist in zip(results["documents"][0], results["distances"][0]):
            # print(f"{1-dist:.4f}: {doc}")
        
        # print(f"\nContext:\n{context}")
        answer = rag.generate_answer(qa["question"], context)
        print(f"\nGenerated Answer: {answer}")
        print(f"\nExpected references:")
        for ref in qa['references']:
            print(f"- {ref}")
        # rag.bleu_scores(answer, qa["references"])

        