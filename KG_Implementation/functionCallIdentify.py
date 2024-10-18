import openai
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# function schemas
functions = [
    {
        "name": "extract_data",
        "description": "Extracts nodes and relationships from the Lean file.",
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
                },
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
                },
            },
            "required": ["nodes", "relationships"],
        },
    },
]


file_path = '../mathlib/src/linear_algebra/matrix/determinant.lean'

with open(file_path, 'r', encoding="utf-8") as file:
        lean_code = file.read()
        
        
# prompt with lean file
prompt = f"""
You are a parser that extracts nodes and relationships from Lean 4 mathlib files.

Please perform the following tasks on the provided Lean file content:

1. Extract all definitions, lemmas, theorems, and abbreviations. For each node, include its type ('def', 'lemma', 'theorem', 'abbreviation') and its name.

2. For each node, identify relationships based on the following categories:
   - Dependencies (Dep): One node depends on another.
   - Equivalence (Equ): Two nodes represent the same concept under different names.
   - Affiliation (Aff): One node is part of or contained in another node.
   - Opposite (Ant): Two nodes are conceptual opposites.
   - Synonym (Syn): Two nodes have similar meanings.
   - Has Property (Pro): One node describes a property of another.

3. Use the `extract_data` function to return the results in structured JSON format.

Lean file content:
{lean_code}
"""

# API call with function defs
response = openai.ChatCompletion.create(
    model='gpt-4o',
    messages=[
        {"role": "user", "content": prompt}
    ],
    functions=functions,
    function_call="auto",
    temperature=0,
)

nodes = []
relationships = []

# extract function call response
message = response['choices'][0]['message']
if 'function_call' in message:
    function_name = message['function_call']['name']
    arguments = message['function_call']['arguments']
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON arguments:")
        print(arguments)
        print(f"Error: {e}")
    else:
        # print outputted JSON for validation
        print(f"Function called: {function_name}")
        print("Outputted JSON:")
        print(json.dumps(data, indent=2))
        
        # extract nodes and relationships from data
        nodes = data.get('nodes', [])
        relationships = data.get('relationships', [])

        # build graph if both nodes and relationships are available
        if 'nodes' in data or 'relationships' in data:
            # directed graph
            G = nx.DiGraph()
            
            # add nodes
            for node in nodes:
                G.add_node(node['name'], node_type=node['node_type'])

            # add edges based on relationships
            for rel in relationships:
                from_node = rel['from_node']
                to_node = rel['to_node']
                relationship_type = rel['relationship_type']
                
                # ensure 'from_node' is in the graph with 'node_type'
                if not G.has_node(from_node):
                    G.add_node(from_node, node_type='unknown')
                
                # ensure 'to_node' is in the graph with 'node_type'
                if not G.has_node(to_node):
                    G.add_node(to_node, node_type='unknown')
                
                # add edge
                G.add_edge(from_node, to_node, label=relationship_type)

            # create node labels, providing a default for missing 'node_type'
            node_labels = {
                node: f"{node}\n({G.nodes[node].get('node_type', 'unknown')})"
                for node in G.nodes()
            }
            
            # draw graph
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=0.3)
            node_labels = {node: f"{node}\n({G.nodes[node]['node_type']})" for node in G.nodes()}
            nx.draw(G, pos, labels=node_labels, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
            
            # add edge labels for relationships
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.show()
else:
    # handle case where model did not use function calling
    print("The model did not return a function call.")
    print("Response content:")
    print(message.get('content', ''))


