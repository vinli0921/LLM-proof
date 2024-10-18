import requests
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dotenv import load_dotenv

load_dotenv()

def read_lean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except UnicodeDecodeError:
        print(f"Error decoding file: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_theoremNames(content):
    # theoremName = r'theorem\s+(\w+)\s'
    theoremName = r'(?:theorem|lemma|def)\s+([a-zA-Z0-9_₀₁₂₃₄₅₆₇₈₉]+)\s'
    theoremNames = re.findall(theoremName, content, re.DOTALL)
    return theoremNames

def analyze_lean_file(api_key, model, file_path):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Read the content of the Lean file
    with open(file_path, 'r', encoding="utf-8") as file:
        lean_code = file.read()
    
        #Define the prompt 
    #     prompt = (
    #     "Analyze the following Lean file code and identify nodes and their related references or dependencies using the data of node names provided:\n\n"
    #     f"{lean_code}\n\n"
    #     "Using the following data of theorem, lemma, and function (def) names " + f"{extract_theoremNames(lean_code)}" + ", list out each name with the following syntax: 'Node: name'" +
    #     "Underneath each node, in a sublist, list all other nodes that have a relationship with the node using one of the following relations: " + 
    #     "Dependencies (Dep): Two node have sequential dependence in the learning process. Example: Learning addition helps us understand the definition of subtraction. Dep (addition, subtraction). " + 
    #     "Affilitation (Aff): Two node have containing and be contained relations. Example: An isoceles triangle is a special kind of triangle. Aff (isoceles triangle, triangle). " + 
    #     "Equivalence (Equ): Two entities with different names but the same meaning. Example: Orthogon belongs to plane geometry, which is also called a rectangle. Equ (orthogon, rectangle). " +
    #     "Opposite (Ant): Two node are conceptually opposite. Example: If a number is even, it must not be odd. Ant (even, odd). " + 
    #     "Synonyms (Syn): Two node have similar meanings. Example: Circles are both axial symmetric and centrally symmetric figures. Syn (axial symmetric, centrally symmetric). " +
    #     "Has Properties (Pro): One node is a property of another. Example: The area of a parallelogram is base times height. Pro (area, parallelogram). The list of relations ends here. " +
    #     "Only list out relations found in the proof. Similarly named nodes do not have a relationship if their proofs do not reference each other. For example: node mk_coe and node coe_mk are independent, despite their names bothing being similar. If a node does not have any references in its proof, still list out the node name, but do not add any additional information below it. Do not add any additional text before or after the results. " + 
    #     "Do not add '- none' if there are nodes already listed in the sublist" +
    #     "An example would be: " + 
    #     "Node: semiconj " + 
    #     "   Dep (semiconj, map_add_const)" +
    #     "Node: coe_mk " +
    #     " "
    # )
        

    prompt = ("You are given a Lean file " + f"{lean_code}" +  " that defines mathematical concepts and theorems and a list of theorem names " +  f"{extract_theoremNames(lean_code)}" + " that will be nodes. Your goal is to: \n\n" + 
    "Understand the syntax of the Lean file to extract information about each node. \n\n" +
    "Store each theorem as a node with relevant attributes. \n\n" +
    "Extract the relationships between the nodes and label the type of relationship. \n\n" +
    "Storing Nodes: \n\n" +
    "For each key node provided: \n\n" +
    "Identify the name and type. A node can be a theorem, lemma, or definition which is denoted by the preceeding code to the name. \n\n" +
    "Capture attributes, such as the parameters or the context in which the concept/theorem is defined (e.g., if it’s under a particular section or depends on a specific algebraic structure). \n\n" +
    "Store the extracted theorem, lemma, or definition as a node in the form: \n\n" +
    "Node(name, type, attributes). When denoting the type, use t for theorem, l for lemma, and d for definition. \n\n" +
    "Example for a theorem: \n\n" +
    "Node(coe_inj, t, (type: equality, context: Field R, Semiring A, proof: injective eq_iff)) \n\n" +
    "Scraping Relationships: \n\n" +
    "For each node pair, extract relationships based on the following categories: \n\n" +
    "Dependencies (Dep): One node depends on another for understanding or derivation. Example: Dep(coe_inj, algebraMap_injective) \n\n" +
    "Equivalence (Equ): Two nodes represent the same concept under different names. Example: Equ(ratCast, algebra_ratCast) \n\n" +
    "Affiliation (Aff): One node is a part of or contained in another node. Example: Aff(algebraMap, Field) \n\n" +
    "Opposite (Ant): Two nodes are opposites conceptually. Example: Ant(zero, one) \n\n" +
    "Synonym (Syn): Two nodes have similar meanings. Example: Syn(map, transform) \n\n" +
    "Has Property (Pro): One node describes a property or characteristic of another. Example: Pro(injective, algebraMap) \n\n" +
    "Store each relationship in the format: \n\n" +
    "relationship(node_x, node_y) \n\n" +
    "The final output should consist of nodes and relationships: \n\n" +
    "Nodes: A list of nodes identified in the file, each with its name, type, and attributes. \n\n" +
    "Relationships: A list of relationships between two nodes, categorized by type. \n\n" +
    "Do not list any additional information before or after the final output. Do not include additional symbols before or after the nodes and relationships."
    )
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,  # Adjust based on the expected length of the response
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    
    # Extract the answer from the response
    if 'choices' in response_data:
        answer = response_data['choices'][0]['message']['content']
        return answer
    return None

# CHANGE API_KEY, MODEL, FILE_PATH
api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4o"  # or gpt-4, depending on your model
file_path = '../mathlib/src/linear_algebra/matrix/determinant.lean'
file_content = read_lean_file(file_path)
# = "mathlibtest/.lake/packages/mathlib/Mathlib/Algebra/AddConstMap/Basic.lean"

def run_code():
    answer = analyze_lean_file(api_key, model, file_path)
    print("Analysis Result:\n", answer)

    # theoremNames = extract_theoremNames(file_content)
    # for name in theoremNames:
    #     print(f"{name}")

run_code()