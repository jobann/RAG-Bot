from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration
from chromadb import PersistentClient
import numpy as np, string

class RAG_QA:
    def __init__(self):
        self.documents = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
        self.generator = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")


    def load_documents(self):
        with open("./data/paragraphs.txt", "r", encoding="utf-8") as f:
            paragraphs = f.read().split("\n\n")

        # for para in paragraphs:
            # self.documents.extend(sent_tokenize(para))

        for para in paragraphs:
            sents = sent_tokenize(para)
            chunk = ""
            for sent in sents:
                if len(chunk.split()) + len(sent.split()) <= 150:
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

        # Simple prompt due to token limit
        input_text = f"question: {question} context: {context}"
       
        print(f"***\n\n{input_text}\n\n")
    

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        output_ids = self.generator.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=150, min_length=10, num_beams=8, no_repeat_ngram_size=1, early_stopping=True, length_penalty=0.8)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def bleu_scores(self, answer):

        references = [
           "With awe, but some fishermen grow angry when they learn Ana knows the secret grounds."
           ]

        ans_tokens = [w for w in word_tokenize(answer.lower()) if w not in string.punctuation]
        ref_tokens = [[w for w in word_tokenize(ref) if w not in string.punctuation] for ref in references]
        smoothie = SmoothingFunction().method4
        scores = [
            sentence_bleu(ref_tokens, ans_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
            sentence_bleu(ref_tokens, ans_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        ]
        print(f"Expected answer: \n{' '.join(references)}\n\nBleu Score:")
        for i, score in enumerate(scores, 1):
            print(f"Bleu{i}: {score:.2f}")

if __name__ == "__main__":
    rag = RAG_QA()
    # rag.load_documents()
    # doc_embeds = rag.embed_documents()
    # collection = rag.create_collection(doc_embeds)
    # question = "How do the villagers react to Santiago’s giant tuna catch?"
    # context, results = rag.retrieve_context(question, collection)
    # print(f"\nRetrieved Sentences:")
    # for doc, dist in zip(results["documents"][0], results["distances"][0]):
    #     print(f"{1-dist:.4f}: {doc}")
    # print(f"\nContext:\n{context}")
    question= "Hello"
    context =""
    answer = rag.generate_answer(question, context)
    print(f"\n\nAnswer: {answer}")
    rag.bleu_scores(answer)