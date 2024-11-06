import os
import openai
import re
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import tiktoken
import json

from RAG.Neo4jVectorRetriever import Neo4jVectorRetriever

from prover.lean.verifier import Lean4ServerScheduler


load_dotenv()

# Initialize OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key

# Initialize Neo4j driver
uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
driver = GraphDatabase.driver(uri, auth=(user, password))

# Initialize Lean 4 server scheduler
lean4_scheduler = Lean4ServerScheduler(
    max_concurrent_requests=1, 
    timeout=300, 
    memory_limit=10, 
    name='verifier'
)


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

# Problem: 
{question}

# Informal proof: ...

# Formal proof in Lean 4: 
```lean4 
...
```
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
    request_id_list = lean4_scheduler.submit_all_request([re.search(r'```[\w]*\n(.*?)\n```', answer, re.DOTALL).group(1)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print(outputs_list[0])
    return (outputs_list[0]['pass']) == True and (outputs_list[0]['complete'] == True), outputs_list[0]

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
                
            # Sort neighbors by similarity and take top k
            neighbor_similarities = []
            for neighbor in neighbors:
                neighbor_embedding = neighbor['embedding']
                if neighbor_embedding is None:
                    continue
                similarity = cosine_similarity(query_embedding, neighbor_embedding)
                neighbor_similarities.append((similarity, neighbor))
                print(f"Similarity at depth {depth}: {similarity}")
            
            # Sort by similarity and take top k
            neighbor_similarities.sort(reverse=True, key=lambda x: x[0])
            top_neighbors = neighbor_similarities[:top_k]
            
            if not top_neighbors:
                print(f"No suitable neighbors found at depth {depth}")
                break
                
            # Add content from all top k neighbors to context
            for similarity, neighbor in top_neighbors:
                content = neighbor.get('content', '')
                context += content + "\n\n"
                print(f"Adding content from neighbor with similarity {similarity}")
            
            # Use the most similar neighbor for next iteration
            best_similarity, best_neighbor = top_neighbors[0]
            current_node_id = best_neighbor['id']
            print(f"Selected best neighbor with similarity {best_similarity} for next iteration")
            
            print(f"Context at depth {depth}: {context}")
            
        # Generate answer with accumulated context
        answer = generate_answer(context, query)
        print(f"Answer at depth {depth}:")
        print(answer)
        
        # Check if the answer is acceptable
        passes, lean_code = is_answer_acceptable(answer)
        if passes:
            print("Answer accepted.")
            return passes, lean_code

    return passes, None

# Main execution
user_query = "Prove this statement: Suppose that a, b, and n are whole numbers. If n does not divide a times b, then n does not divide a and n does not divide b."

with open('datasets/minif2f.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

test_data = [entry for entry in data if entry.get('split') == 'test']

print(f"Total test cases: {len(test_data)}")


results = []

# for entry in test_data:
#     name = entry['name']
#     informal_prefix = entry.get('informal_prefix', '')
#     formal_statement = entry.get('formal_statement', '')
#     header = entry.get('header', '')

#     # Combine header and formal_statement
#     input = f"{header}\n\n{informal_prefix}\n\n{formal_statement}"

#     # Check if the Lean code passes verification
#     passes, lean_code = answer_user_query(input, driver)

#     # Record the result
#     result = {
#         'name': name,
#         'passed': passes,
#         'lean_code': lean_code
#     }
#     results.append(result)

#     # Optional: Print progress
#     print(f"Processed {name}: {'Passed' if passes else 'Failed'}")

for entry in test_data:
    print(entry)
    name = entry['name']
    header = entry.get('header', '')
    informal_prefix = entry.get('informal_prefix', '')
    formal_statement = entry.get('formal_statement', '')
    goal = entry.get('goal', '')
    

    # Combine header and formal_statement
    input = """
Given this problem statement: 
{informal_prefix}
Complete the following Lean 4 code:
{header}
{formal_statement}
to reach this goal: 
{goal}
    """.format(informal_prefix=informal_prefix, header=header, formal_statement=formal_statement, goal=goal)

    max_attempts = 3  # Total number of attempts per entry
    attempt = 0
    passes = False
    output = ""

    while attempt < max_attempts and not passes:
        try:
            print(f"Processing {name}, Attempt {attempt + 1}")
            # Generate the answer
            answer = generate_answer(output, input)
            # Check if the Lean code passes verification
            passes, output = is_answer_acceptable(answer)
            if passes:
                print(f"Attempt {attempt + 1} succeeded for {name}.")
            else:
                print(f"Attempt {attempt + 1} failed for {name}. Retrying...")
        except Exception as e:
            print(f"Error processing {name} on attempt {attempt + 1}: {e}")
            print(f"Retrying {name}...")
        finally:
            attempt += 1

    # Record the result
    result = {
        'name': name,
        'passed': passes,
        'lean_code': output["verified_code"]
    }
    results.append(result)

    # Optional: Print progress
    print(f"Processed {name}: {'Passed' if passes else 'Failed'}")



try:
    # Calculate statistics first
    num_passed = sum(1 for result in results if result['passed'])
    num_failed = len(results) - num_passed

    print(f"Total test cases: {len(results)}")
    print(f"Passed: {num_passed}")
    print(f"Failed: {num_failed}")

    # Write results to file
    with open('4o-mini_minif2f_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

finally:
    # Clean up resources
    lean4_scheduler.close()
    driver.close()
# answer = answer_user_query(user_query, driver, max_depth=3, top_k=5)
# print("Final Answer:")
# print(answer)
