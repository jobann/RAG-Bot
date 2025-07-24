import requests

def read_paragraph(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph_text = file.read()
    
    print("Loaded Paragraph:\n")
    print(f"{paragraph_text}\n\n")
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