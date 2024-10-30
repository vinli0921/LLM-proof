import openai
import csv
import os
from dotenv import load_dotenv
from tiktoken import get_encoding

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize encoding for token counting
encoding = get_encoding("cl100k_base")  # Use appropriate encoding for your model

# Define maximum tokens allowed by the model
MAX_TOKENS = 8191  # Adjust based on the specific model's limit
EMBEDDING_MODEL = "text-embedding-3-large"  # Replace with your embedding model (3072)

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def split_text(text, max_tokens):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def generate_embeddings(input_csv, output_csv):
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'embedding'])
        
        # Skip the header row
        next(reader, None)
        
        count = 1
        for row in reader:
            node_id, type, title, name, content = row
            combined_text = f"{title} {content}"
            
            # Check token length
            token_count = len(encoding.encode(combined_text))
            if token_count > MAX_TOKENS:
                print(f"{count}. Text too long for Node {node_id} : {name}, splitting into chunks.")
                chunks = split_text(combined_text, MAX_TOKENS // 2)  # Split into smaller chunks
                embeddings = []
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    embeddings.append(embedding)
                # Optionally, aggregate embeddings (e.g., average)
                aggregated_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
                writer.writerow([node_id, aggregated_embedding])
            else:
                print(f"{count}. Generating embedding for Node {node_id} : {name}")  # debugging statement
                embedding = get_embedding(combined_text)
                writer.writerow([node_id, embedding])
            
            count += 1

# Example function to get embedding size
def embedding_size():
    response = openai.embeddings.create(
        input="Hello, World!",
        model=EMBEDDING_MODEL
    )
    return len(response.data[0].embedding)

if __name__ == "__main__":
    input_csv = '../nodes.csv'  # path to input CSV file
    output_csv = 'embeddings.csv'  # path to output CSV file
    print(f"Embedding size: {embedding_size()}")  # debugging statement
    # generate_embeddings(input_csv, output_csv)
