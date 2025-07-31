import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from chromadb import PersistentClient
import numpy as np, string
import requests


class Read_DrawIO:
    def detect_diagram_type(self, styles):
        # Count occurrences of keywords in styles
        style_counter = Counter()
        for style in styles:
            if 'uml' in style:
                style_counter['uml'] += 1
            if 'process' in style or 'decision' in style or 'terminator' in style:
                style_counter['flowchart'] += 1
            if 'swimlane' in style:
                style_counter['flowchart'] += 1  # Swimlanes common in flowcharts
            if 'entity' in style or 'table' in style:
                style_counter['er'] += 1  # Entity-Relationship diagrams
            if 'cloud' in style or 'router' in style or 'server' in style:
                style_counter['network'] += 1

        if not style_counter:
            return "Unknown"
        
        # Find the highest count category
        diagram_type, count = style_counter.most_common(1)[0]

        # Map to human-readable name
        mapping = {
            'uml': "UML Diagram",
            'flowchart': "Flowchart",
            'er': "ER Diagram",
            'network': "Network Diagram"
        }
        return mapping.get(diagram_type, "Unknown")

    def parse_drawio(self, xml_string):
        tree = ET.ElementTree(ET.fromstring(xml_string))
        root = tree.getroot()

        Class = {}
        members = defaultdict(list)
        relationships = []
        id_to_name = {}
        all_styles = []

        for cell in root.iter('mxCell'):
            cell_id = cell.attrib.get('id')
            value = cell.attrib.get('value', '').strip()
            style = cell.attrib.get('style', '')
            parent_id = cell.attrib.get('parent')
            vertex = cell.attrib.get('vertex')
            edge = cell.attrib.get('edge')

            if style:
                all_styles.append(style)

            # Only process vertices that have a value
            if vertex == "1" and value:
                if "swimlane" in style:
                    # Component / Class
                    Class[cell_id] = value
                else:
                    # Member (attribute or method)
                    members[parent_id].append(value)
                id_to_name[cell_id] = value  # Map all named vertices

            # Relationships (edges between nodes)
            elif edge == "1":
                source = cell.attrib.get('source')
                target = cell.attrib.get('target')
                if source and target:
                    relationships.append((source, target))

        diagram_type = self.detect_diagram_type(all_styles)

        return Class, members, relationships, id_to_name, diagram_type


    def get_Class_and_relationships(self, Class, members, relationships, id_to_name, diagram_type):

       xmlInfo = f"## Diagram Analysis\n"
       xmlInfo += f"**Detected Diagram Type:** {diagram_type}\n\n"
       
       xmlInfo += "### Components and their Groupings:\n"
       for comp_id, comp_name in Class.items():
            xmlInfo += f"- **Component:** {comp_name}\n"
            for m in members.get(comp_id, []):
                xmlInfo += f"  - Member: {m}\n"
            xmlInfo += "\n"
            
       xmlInfo += "### Relationships Between Components:\n"
       for source_id, target_id in relationships:
            source_name = id_to_name.get(source_id, f"(unknown:{source_id})")
            target_name = id_to_name.get(target_id, f"(unknown:{target_id})")
            xmlInfo += f"- {source_name} → {target_name}\n"

       print(xmlInfo)

       return xmlInfo

        # print(f"Detected diagram type: {diagram_type}\n")
        # print("Class and their members:\n")
        # for comp_id, comp_name in Class.items():
        #     print(f"Class: {comp_name}")
        #     for m in members.get(comp_id, []):
        #         print(f"\t{m}")
        #     print()

        # print("Relationships between Class:\n")
        # for source_id, target_id in relationships:
        #     source_name = id_to_name.get(source_id, f"(unknown:{source_id})")
        #     target_name = id_to_name.get(target_id, f"(unknown:{target_id})")
        #     print(f"  {source_name} --> {target_name}")







class RAG_QA:
    
    def get_suggestion(self, xmlInfo):

        xmlInfo += """
        #### Task:
        You are a cloud architecture expert. Analyze the above AWS-based architecture and provide clear, concise, and actionable suggestions to improve the design based on the following goals:

        -  **Performance Optimization**
        -  **Cost Efficiency**
        -  **Scalability**
        -  **Security Best Practices**
        -  **Monitoring and Observability**

        For each suggestion, briefly explain it's beneficial in the context of this specific diagram.
        Return your suggestions in a clean format without '*'.
        """
        return self.get_response_from_llama(xmlInfo)
    
       
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





# Usage: read from file or string
if __name__ == "__main__":

    file_path = "./data/AzureApplication.drawio"
    with open(file_path, "r", encoding="utf-8") as f:
        xml_string = f.read()

    read_drawio = Read_DrawIO()
    Class, members, relationships, id_to_name, diagram_type = read_drawio.parse_drawio(xml_string)
    xmlInfo = read_drawio.get_Class_and_relationships(Class, members, relationships, id_to_name, diagram_type)

    rag_qa = RAG_QA()
    suggestion = rag_qa.get_suggestion(xmlInfo)

    print(f"\n\nSuggestions to improve the architecture:\n{suggestion}")

