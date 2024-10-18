import requests
import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
import os

import json

load_dotenv()


functions = [
    {
        "name": "extract_nodes",
        "description": "Extracts definitions, lemmas, theorems, and abbreviations from the Lean file.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_type": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["node_type", "name"],
                    },
                }
            },
            "required": ["nodes"],
        },
    },
    {
        "name": "extract_relationships",
        "description": "Extracts relationships between nodes based on the specified categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relationship_type": {"type": "string"},
                            "from_node": {"type": "string"},
                            "to_node": {"type": "string"},
                        },
                        "required": ["relationship_type", "from_node", "to_node"],
                    },
                }
            },
            "required": ["relationships"],
        },
    },
]


# Define regex patterns for definitions, lemmas, and theorems
definition_pattern = re.compile(r'^(def|abbreviation)\s+(\S+)\s*[:\(]', re.MULTILINE)
lemma_pattern = re.compile(r'^(lemma|theorem|@[^\n]*\s+(lemma|theorem))\s+(\S+)\s*', re.MULTILINE)
comment_pattern = re.compile(r'/-(?:.|\n)*?-/')  # To remove multiline comments

def analyze_lean_file(api_key, model, file_path):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Read the content of the Lean file
    with open(file_path, 'r', encoding="utf-8") as file:
        lean_code = file.read()
    
    # Extract definitions
    definitions = definition_pattern.findall(lean_code)
    definitions = [name for (_, name) in definitions]

    # Extract lemmas and theorems
    lemmas = lemma_pattern.findall(lean_code)
    lemmas = [name for (_, _, name) in lemmas]

    # Combine all nodes
    nodes = definitions + lemmas
    
    # Define the prompt for the ChatGPT API
    prompt = (
            "I have the following Lean 4 mathlib file content:"

            f"{lean_code}"

            "Please perform the following tasks:"

            "1. Extract all definitions, lemmas, and theorems, and list them as nodes. Precede each node with its type (e.g., 'def', 'lemma', 'theorem', 'abbreviation')."

            "2. For each node, identify and list the relationships based on the following categories:"

            "- Dependencies (Dep): One node depends on another for understanding or derivation."
            "- Equivalence (Equ): Two nodes represent the same concept under different names."
            "- Affiliation (Aff): One node is a part of or contained in another node."
            "- Opposite (Ant): Two nodes are opposites conceptually."
            "- Synonym (Syn): Two nodes have similar meanings."
            "- Has Property (Pro): One node describes a property or characteristic of another."

            "3. Present the output in a structured format, such as:"

            "Nodes:"
            "- def det_row_alternating"
            "- abbreviation det"
            "- lemma det_apply"
            "..."

            "Relationships:"
            "- Dep(det, det_row_alternating)"
            "- Equ(det_apply, det_apply)"
            "..."

            "Please make sure to include all relevant nodes and relationships based on the provided Lean file content."
        )
    #     f"You are given a Lean file {lean_code} that defines mathematical concepts and theorems. Your goal is to:\n\n"
    #     "Understand the syntax of the Lean file to extract key concepts and theorems.\n\n"
    #     "Store each concept or theorem as a node with relevant attributes.\n\n"
    #     "Extract the relationships between the nodes (concepts and theorems) and label the type of relationship.\n\n"
    #     "Instructions:\n\n"
    #     "Understanding Syntax:\n\n"
    #     "Concepts: A concept is typically defined as a variable, type, structure, or function.\n\n"
    #     "Theorems: A theorem is a mathematical statement proven within the Lean file. These usually start with keywords like theorem, lemma, or definition.\n\n"
    #     "Transformations: Identify any conversions or operations such as injections, divisions, exponentiations, and their mappings.\n\n"
    #     "Storing Nodes:\n\n"
    #     "For each key concept or theorem:\n\n"
    #     "Identify the name and type (concept/theorem).\n\n"
    #     "Capture attributes, such as the parameters or the context in which the concept/theorem is defined (e.g., if it's under a particular section or depends on a specific algebraic structure).\n"
    #     "Store the extracted concept or theorem as a node in the form:\n\n"
    #     "Node(name, type, attributes)\n\n"
    #     "Example for a theorem:\n\n"
    #     "Node(coe_inj, theorem, [type: equality, context: Field R, Semiring A, proof: injective.eq_iff])\n\n"
    #     "Scraping Relationships:\n\n"
    #     "For each node pair (concept/theorem), extract relationships based on the following categories:\n\n"
    #     "Dependencies (Dep): One node (concept/theorem) depends on another for understanding or derivation. Example: Dep(coe_inj, algebraMap_injective)\n\n"
    #     "Equivalence (Equ): Two nodes represent the same concept under different names. Example: Equ(ratCast, algebra_ratCast)\n\n"
    #     "Affiliation (Aff): One node is a part of or contained in another node. Example: Aff(algebraMap, Field)\n\n"
    #     "Opposite (Ant): Two nodes are opposites conceptually. Example: Ant(zero, one)\n\n"
    #     "Synonym (Syn): Two nodes have similar meanings. Example: Syn(map, transform)\n\n"
    #     "Has Property (Pro): One node describes a property or characteristic of another. Example: Pro(injective, algebraMap)\n\n"
    #     "Store each relationship in the format:\n\n"
    #     "relationship(node_x, node_y)\n\n"
    #     "Final Output:\n\n"
    #     "Nodes: A list of nodes identified in the file, each with its name, type, and attributes.\n\n"
    #     "Relationships: A list of relationships between these nodes, categorized by type\n\n"
    # )
    # print("Nodes:")
    # for node in nodes:
    #     print(f"- {node}")
    
    response = openai.ChatCompletion.create(
        model='gpt-4o',  # Ensure you're using a model that supports function calling
        messages=[
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call="auto",
        temperature=0,
    )

    # Extract the function call response
    if 'function_call' in response['choices'][0]['message']:
        function_name = response['choices'][0]['message']['function_call']['name']
        arguments = response['choices'][0]['message']['function_call']['arguments']
        data = json.loads(arguments)
        if function_name == 'extract_nodes':
            nodes = data['nodes']
        elif function_name == 'extract_relationships':
            relationships = data['relationships']
            
    # data = {
    #     "model": model,
    #     "messages": [{"role": "user", "content": prompt}],
    #     "max_tokens": 2500,  # Adjust based on the expected length of the response
    # }
    
    # response = requests.post(url, headers=headers, json=data)
    # response_data = response.json()
    
    # Extract the answer from the response
    # if 'choices' in response_data:
    #     answer = response_data['choices'][0]['message']['content']
    #     return answer
    # return None

# CHANGE API_KEY, MODEL, FILE_PATH
api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4o"  # or gpt-4, depending on your model
file_path = '../mathlib/src/linear_algebra/matrix/determinant.lean'

answer = analyze_lean_file(api_key, model, file_path)
print("Analysis Result:\n", answer)

# Adjusted regex pattern
relationship_pattern = re.compile(
    r'^\s*-?\s*\**\s*(Dep|Equ|Aff|Ant|Syn|Pro)\s*\(\s*([^)]+?)\s*\)\s*\**',
    re.MULTILINE
)

matches = relationship_pattern.findall(answer)

adjacency_list = defaultdict(list)

# Iterate over the matches and parse the nodes
for match in matches:
    relationship, nodes_str = match
    # Split the nodes by comma and strip whitespace
    nodes = [node.strip() for node in nodes_str.split(',')]
    node1 = nodes[0]
    # For relationships involving multiple nodes, create edges accordingly
    for node2 in nodes[1:]:
        adjacency_list[node1].append((relationship, node2))

# Convert defaultdict to regular dict for printing (optional)
adjacency_list = dict(adjacency_list)

# Display the adjacency list
for node, edges in adjacency_list.items():
    print(f"Node: {node} Edges: {edges}")

# Initialize directed graph
G = nx.DiGraph()

# Add nodes and edges from the adjacency list
for node, edges in adjacency_list.items():
    for edge_type, related_node in edges:
        G.add_edge(node, related_node, label=edge_type)

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)

# Add edge labels for relationships
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.show()