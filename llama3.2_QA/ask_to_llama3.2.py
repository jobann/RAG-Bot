import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import string

def read_paragraph(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph_text = file.read()
    
    print("Loaded Paragraph:\n")
    # print(f"{paragraph_text}\n\n")
    return paragraph_text

def get_response_from_llama(query_prompt: str, model: str="llama3.2"):
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
    paragraph_text = read_paragraph("./data/paragraphs.txt")

    while True:
        query_prompt = input(f"\nAsk your question (or type 'exit' to quit): ").strip()
        if(query_prompt.lower() == "exit" or query_prompt.lower() == "quit"):
            print("Goodbye!")
            break;
    
        prompt = f"""You are a helpful agent. Read the following paragraph and use the provided question to generate a clear, relevant, and informative answer based only on the content of the paragraph.
                Paragraph:
                {paragraph_text}
                Question:
                {query_prompt}
                Now, provide a concise and accurate answer using only the information in the paragraph."""
        response = get_response_from_llama(prompt)
        print(f"{response}")

        # Calculating BLEU score
        answer_tokens = [word for word in word_tokenize(response.lower()) if word not in string.punctuation]

        references = [
           "To study declining fish populations and document changes in marine ecosystems."
            ]

        references_tokens = [
            [word for word in word_tokenize(sentence) if word not in string.punctuation] 
            for sentence in references]

        smoothie = SmoothingFunction().method4

        bleu1 = sentence_bleu(references_tokens, answer_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(references_tokens, answer_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu(references_tokens, answer_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(references_tokens, answer_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        print(f"Bleu1: {bleu1:.2f}")
        print(f"Bleu2: {bleu2:.2f}")
        print(f"Bleu3: {bleu3:.2f}")
        print(f"Bleu4: {bleu4:.2f}")