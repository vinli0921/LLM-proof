import os
import openai
import numpy as np
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import tiktoken
import tempfile
import subprocess

from Neo4jVectorRetriever import Neo4jVectorRetriever

# Initialize OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key

# Initialize Neo4j driver
uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
driver = GraphDatabase.driver(uri, auth=(user, password))

# Initialize LLM
llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4o-mini', temperature=0)

# Answer Generation Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language (indicated by ``Informal proof:''), and a translated formal solution in Lean 4 (indicated by ``Formal proof in Lean 4:''). Please note that the informal proof and the formal proof need to be identical."
    ),
    (
        "system",
        "Context:\n{context}"
    ),
    (
        "human",
        """You must respond in the following format: 

# Problem: {question}

# Informal proof: ...

# Formal proof in Lean 3: ...

"""
    )
])
answer_chain = answer_prompt | llm | StrOutputParser()

def generate_answer(context, question):
    response = answer_chain.invoke({"context": context, "question": question})
    return response.strip()

def compile_context(documents, max_tokens=128000):
    # Combine and truncate context to fit token limit
    context = "\n\n".join([doc.page_content for doc in documents])
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    tokens = encoding.encode(context)
    if len(tokens) <= max_tokens:
        return context
    else:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_answer_acceptable(answer):
    # Extract Lean code enclosed in ```lean ... ```
    import re
    pattern = r'```lean(.*?)```'
    matches = re.findall(pattern, answer, re.DOTALL)
    if not matches:
        print("No Lean code found in the answer.")
        return False
    lean_code = matches[0].strip()
    # Write the Lean code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as temp_file:
        temp_file.write(lean_code)
        temp_filename = temp_file.name
    try:
        # Run Lean on the temporary file
        result = subprocess.run(['lean', temp_filename], capture_output=True, text=True)
        if result.returncode == 0:
            print("Lean code ran successfully without errors.")
            return True
        else:
            print("Lean code has errors:")
            print(result.stderr)
            return False
    finally:
        # Clean up the temporary file
        os.remove(temp_filename)

def answer_user_query(query, driver, max_depth=3, top_k=5):
    retriever = Neo4jVectorRetriever(driver=driver, top_k=top_k)
    
    # Compute query_embedding
    query_embedding = retriever.get_query_embedding(query)
    
    context = ""
    current_node_id = None

    for depth in range(max_depth + 1):
        if depth == 0:
            # Get the top node similar to the query
            current_node_id, content = retriever.get_top_node(query_embedding)
            if current_node_id is None:
                print("No relevant node found.")
                break
            context += content + "\n\n"
            print(f"Context at depth {depth}: {content}")
        else:
            # Get k neighbors of the current node
            neighbors = retriever.get_neighbors(current_node_id, top_k)
            if not neighbors:
                print(f"No neighbors found at depth {depth}")
                break
            # Compute similarity between query_embedding and each neighbor's embedding
            max_similarity = -1
            best_neighbor = None
            for neighbor in neighbors:
                neighbor_embedding = neighbor['embedding']
                if neighbor_embedding is None:
                    continue  # Skip if no embedding
                similarity = cosine_similarity(query_embedding, neighbor_embedding)
                print(f"Similarity at depth {depth}: {similarity}")
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_neighbor = neighbor
            if best_neighbor is None:
                print(f"No suitable neighbor found at depth {depth}")
                break
            # Update current_node_id and context
            current_node_id = best_neighbor['id']
            content = best_neighbor.get('content', '')
            context += content + "\n\n"
            print(f"Context at depth {depth}: {context}")
        # Generate answer
        answer = generate_answer(context, query)
        print(f"Answer at depth {depth}:")
        print(answer)
        # Check if the answer is acceptable
        if is_answer_acceptable(answer):
            print("Acceptable answer found.")
            break
    return answer

# Main execution
user_query = "Prove this statement: Suppose that a, b, and n are whole numbers. If n does not divide a times b, then n does not divide a and n does not divide b."

answer = answer_user_query(user_query, driver, max_depth=3, top_k=5)

print("Final Answer:")
print(answer)
driver.close()
