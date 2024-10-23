import openai
import csv
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response['data'][0]['embedding']

def generate_embeddings(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'embedding'])
        
        # skip the header row
        next(reader, None)

        for row in reader:
            node_id, type, title, name, content = row
            embedding = get_embedding(title + content)
            writer.writerow([node_id, embedding])

if __name__ == "__main__":
    input_csv = '../nodes.csv'  # path to input CSV file
    output_csv = 'embeddings.csv'  # path to output CSV file
    generate_embeddings(input_csv, output_csv)