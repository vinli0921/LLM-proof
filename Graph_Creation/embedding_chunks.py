import openai
import csv
import os
from dotenv import load_dotenv
from tiktoken import get_encoding
from neo4j import GraphDatabase

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# Initialize encoding for token counting
encoding = get_encoding("cl100k_base")  # Use appropriate encoding for your model

# Define maximum tokens allowed by the model
MAX_TOKENS = 8191  # Adjust based on the specific model's limit
EMBEDDING_MODEL = "text-embedding-3-large"  # Replace with your embedding model (3072)
BATCH_SIZE = 100  # Number of nodes to process at once

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
            
def generate_neo4j_embeddings(output_csv):
    """
    Generate embeddings for nodes in Neo4j database and save to CSV.
    
    Args:
        output_csv (str): Path to output CSV file
    """
    def get_total_nodes(tx):
        result = tx.run("MATCH (n:Node) RETURN count(n) as count")
        return result.single()["count"]
    
    def get_batch_nodes(tx, skip, limit):
        result = tx.run("""
            MATCH (n:Node)
            RETURN n.id as id, n.title as title, n.content as content
            SKIP $skip
            LIMIT $limit
        """, skip=skip, limit=limit)
        return [record for record in result]

    with gds.session() as session:
        # Get total number of nodes
        total_nodes = session.execute_read(get_total_nodes)
        print(f"Total nodes to process: {total_nodes}")

        # Open CSV file for writing
        with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id', 'embedding'])
            
            # Process nodes in batches
            count = 1
            for skip in range(0, total_nodes, BATCH_SIZE):
                # Get batch of nodes
                nodes = session.execute_read(get_batch_nodes, skip, BATCH_SIZE)
                
                for node in nodes:
                    node_id = node["id"]
                    title = node["title"] or ""  # Handle None values
                    content = node["content"] or ""  # Handle None values
                    
                    combined_text = f"{title} {content}".strip()
                    if not combined_text:  # Skip empty nodes
                        print(f"{count}. Skipping empty Node {node_id}")
                        continue

                    # Check token length and generate embedding
                    token_count = len(encoding.encode(combined_text))
                    if token_count > MAX_TOKENS:
                        print(f"{count}. Text too long for Node {node_id}, splitting into chunks.")
                        chunks = split_text(combined_text, MAX_TOKENS // 2)
                        embeddings = []
                        for chunk in chunks:
                            embedding = get_embedding(chunk)
                            embeddings.append(embedding)
                        # Average the embeddings from chunks
                        aggregated_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
                        writer.writerow([node_id, aggregated_embedding])
                    else:
                        print(f"{count}. Generating embedding for Node {node_id}")
                        embedding = get_embedding(combined_text)
                        writer.writerow([node_id, embedding])
                    
                    count += 1
                
                processed = min(skip + BATCH_SIZE, total_nodes)
                print(f"Processed {processed} out of {total_nodes} nodes")

# Example function to get embedding size
def embedding_size():
    response = openai.embeddings.create(
        input="Hello, World!",
        model=EMBEDDING_MODEL
    )
    return len(response.data[0].embedding)

if __name__ == "__main__":
    # input_csv = '../nodes.csv'  # path to input CSV file
    # output_csv = 'embeddings.csv'  # path to output CSV file
    # print(f"Embedding size: {embedding_size()}")  # debugging statement
    # generate_embeddings(input_csv, output_csv)
    output_csv = 'embeddings.csv'
    try:
        print(f"Starting embedding generation...")
        generate_neo4j_embeddings(output_csv)
        print(f"Embedding generation completed. Results saved to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
