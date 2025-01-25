import os
import json
import re
import csv
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from anthropic import Anthropic
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from RAG.Neo4jVectorRetriever import Neo4jVectorRetriever
from prover.lean.verifier import Lean4ServerScheduler

load_dotenv()

@dataclass
class TestCase:
    name: str
    header: str
    informal_prefix: str
    formal_statement: str
    goal: str

@dataclass
class ProofAttempt:
    informal_proof: str
    formal_proof: str
    success: bool
    error_message: Optional[str] = None

class ProofGenerator:
    """Responsible for generating informal mathematical proofs"""
    def __init__(self, model_type="gpt-4o", temperature=0):
        if "gpt" in model_type.lower() or "o1" in model_type.lower():
            self.llm = ChatOpenAI(
                model_name=model_type,
                temperature=temperature,
                openai_api_key=os.environ.get('OPENAI_API_KEY')
            )
            self.use_anthropic = False
        else:
            self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
            self.model_name = model_type
            self.temperature = temperature
            self.use_anthropic = True
            
    def generate_proof(self, context: str, problem: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a mathematics expert focused on generating clear informal proofs."),
            ("user", """Given the following mathematical problem and context, generate a clear and detailed informal proof in natural language.
Do not attempt to formalize the proof yet - focus only on explaining the mathematical reasoning clearly.

The context contains theorems, proofs, and mathematical expressions that may be relevant to solving this problem.
Pay special attention to:
- Similar theorem statements
- Related proof techniques
- Mathematical patterns and structures
- Definitions and axioms used

Context:
{context}

Problem to Prove:
{problem}

Provide your proof in the following format:

# Informal Proof:
[Your natural language proof here]
""")
        ])
        if self.use_anthropic:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_template}]
            )
            return response.content[0].text.strip()
        else:
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({
                "context": context,
                "problem": problem
            }).strip()
            
class PseudocodeGenerator:
    """
    Responsible for converting an informal proof into a structured,
    step-by-step pseudocode-like mathematical outline.
    (Agent #2 in the chain)
    """
    def __init__(self, model_type="gpt-4o", temperature=0):
        if "gpt" in model_type.lower() or "o1" in model_type.lower():
            self.llm = ChatOpenAI(
                model_name=model_type,
                temperature=temperature,
                openai_api_key=os.environ.get('OPENAI_API_KEY')
            )
            self.use_anthropic = False
        else:
            self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
            self.model_name = model_type
            self.temperature = temperature
            self.use_anthropic = True

    def generate_pseudocode(self, informal_proof: str) -> str:
        """
        Given an informal proof, produce a step-by-step pseudocode approach.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a mathematics teaching assistant who creates structured outlines from informal proofs."),
            ("user", """You are given the following informal proof. Convert it into a structured pseudocode approach, step by step.
Format it in a numbered or bullet-list style that details the main logical steps, references to known theorems, etc.

Informal Proof:
{informal_proof}

# Pseudocode:
""")
        ])
        if self.use_anthropic:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_template}]
            )
            return response.content[0].text.strip()
        else:
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({
                "informal_proof": informal_proof
            }).strip()

class AutoFormalizer:
    """Responsible for converting informal proofs to Lean 4 formal proofs"""
    def __init__(self, model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL", temperature=0.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = LLM(model=model_name, max_num_batched_tokens=8192, seed=1, trust_remote_code=True)
        self.temperature = temperature
    
    def formalize_proof(self, header: str, pseudocode: str, informal_proof: str, informal_prefix: str, formal_statement: str, goal: str) -> str:
        # Construct the prompt
        prompt = f"""You are a Lean 4 code generator. Utilize the below information and follow the instructions.
We have:
  HEADER:
{header}

  PSEUDOCODE:
{pseudocode}

  PREFIX:
{informal_prefix}

  STATEMENT:
{formal_statement}

GOAL (optional):
{goal}

INSTRUCTIONS:
1. Output exactly one triple-backtick code block containing valid Lean 4 code.
2. Do not include any text or explanations outside the code block.
3. Make sure it compiles in Lean 4.

Required Format:
# Start
```lean4
<Lean code here>
``` # End
"""
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048,
            top_p=0.1,
            n=1,
        )
        
        # Generate the formal proof using the DeepSeek model
        model_outputs = self.model.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )

        generated_text = model_outputs[0].outputs[0].text

        # Return the formal proof
        return generated_text.strip()

class ThreeAgentProver:
    """ Orchestrates the RAG context retrieval and the 3-step generation process: 
    1) Informal Proof generation (with context) 
    2) Pseudocode generation from Informal Proof 
    3) Formal Lean code generation from Pseudocode 
    Then attempts to verify the resulting Lean code. 
    """ 
    def __init__(
        self, 
        lean4_scheduler, 
        neo4j_uri, 
        neo4j_user, 
        neo4j_password, 
        proof_generator: ProofGenerator, 
        pseudocode_generator: PseudocodeGenerator, 
        auto_formalizer: AutoFormalizer, 
        max_depth=3, 
        top_k=5, 
        max_attempts=3, 
        log_file=None 
    ): 
        self.lean4_scheduler = lean4_scheduler 
        self.proof_generator = proof_generator 
        self.pseudocode_generator = pseudocode_generator 
        self.auto_formalizer = auto_formalizer 
        self.max_depth = max_depth 
        self.max_attempts = max_attempts
        
        # Initialize Neo4j and RAG components
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.retriever = Neo4jVectorRetriever(driver=self.driver, top_k=top_k)
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'three_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        # Adjust the CSV header to store pseudocode
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'problem', 'prompt', 'depth', 'attempt', 'visited_node_ids',
                'informal_proof', 'pseudocode', 'formal_proof', 'passed'
            ])
            
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        """
        Submits the formal proof to the Lean4 verifier and returns (passed, details).
        """
        code_match = re.search(r'```lean4?\n(.*?)\n```', formal_proof, re.DOTALL)
        if code_match:
            code_to_verify = code_match.group(1)
        else:
            # If no code block found, fallback to entire text
            code_to_verify = formal_proof

        request_id_list = self.lean4_scheduler.submit_all_request([code_to_verify])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        print("Verification result:", result)
        return (result['pass'] == True and result['complete'] == True), result
    
    def _log_attempt(
        self, 
        problem: str,
        prompt: str,
        depth: int,
        attempt: int, 
        visited_node_ids: List[str],
        informal_proof: str,
        pseudocode: str,
        formal_proof: str,
        passed: bool
    ):
        """
        Logs a single attempt to the CSV file.
        """
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            informal_escaped = informal_proof.replace('\n', '\\n')
            pseudocode_escaped = pseudocode.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            visited_str = ",".join(visited_node_ids)
            
            writer.writerow([
                problem, prompt, depth, attempt, visited_str,
                informal_escaped, pseudocode_escaped, formal_escaped, passed
            ])

    def _get_rag_context(self, query: str, depth: int) -> str:
        """
        Retrieves relevant context from the Neo4j database up to a specified depth.
        """
        query_embedding = self.retriever.get_query_embedding(query)
        context = ""
        visited_ids = []
        current_node_id = None
        
        for d in range(depth + 1):
            if d == 0:
                current_node_id, content = self.retriever.get_top_node(query_embedding)
                if current_node_id is None:
                    return "", []
                visited_ids.append(str(current_node_id))
                context += f"{current_node_id}:\n{content}\n\n"
            else:
                neighbors = self.retriever.get_neighbors(current_node_id, self.retriever.top_k)
                if not neighbors:
                    break
                
                neighbor_similarities = [
                    (self._cosine_similarity(query_embedding, n['embedding']), n)
                    for n in neighbors if n['embedding'] is not None
                ]
                
                if not neighbor_similarities:
                    break
                    
                neighbor_similarities.sort(reverse=True, key=lambda x: x[0])
                top_neighbors = neighbor_similarities[:self.retriever.top_k]
                
                for _, neighbor in top_neighbors:
                    content = neighbor.get('content', '')
                    node_id = neighbor.get('id')
                    visited_ids.append(str(node_id))
                    context += f"{node_id}:\n{content}\n\n"
                
                current_node_id = top_neighbors[0][1]['id']
        
        return context.strip(), visited_ids

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_test_case(self, test_case: TestCase) -> Dict:
        """
        Executes the 3-step pipeline for a single test case,
        possibly multiple times (max_attempts) across increasing RAG depths.
        """
        for depth in range(self.max_depth + 1):
            print(f"\nTrying at depth {depth}")
            
            # 1) Get RAG context for current depth
            rag_context, visited_node_ids = self._get_rag_context(test_case.informal_prefix, depth)
            if not rag_context:
                print(f"No context found at depth {depth}, moving to next depth")
                continue
            
            # 2) Generate informal proof using RAG context
            informal_proof = self.proof_generator.generate_proof(
                rag_context, 
                test_case.informal_prefix
            )
            
            # Attempt multiple formalization tries
            for attempt in range(self.max_attempts):
                try:
                    print(f"Processing {test_case.name}, Depth {depth}, Attempt {attempt + 1}")
                    
                    # 2.5) Convert informal proof to pseudocode
                    pseudocode = self.pseudocode_generator.generate_pseudocode(informal_proof)
                    
                    # 3) Convert pseudocode to formal Lean code
                    formal_proof = self.auto_formalizer.formalize_proof(
                        test_case.header,
                        pseudocode,
                        informal_proof, 
                        test_case.informal_prefix,
                        test_case.formal_statement,
                        test_case.goal
                    )
                    
                    # Verify with Lean
                    passes, output = self._verify_lean_proof(formal_proof)
                    
                    # Log the attempt
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        visited_node_ids=visited_node_ids,
                        informal_proof=informal_proof,
                        pseudocode=pseudocode,
                        formal_proof=formal_proof,
                        passed=passes
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'visited_node_ids': visited_node_ids,
                            'informal_proof': informal_proof,
                            'pseudocode': pseudocode,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1
                        }
                    
                    # If verification failed, add error context for next attempt to refine
                    if isinstance(output, dict) and 'errors' in output:
                        error_context = "\n".join([error.get('data', '') for error in output['errors']])
                        # Append a note to the existing informal proof, so next attempt can incorporate it
                        informal_proof += f"\nPrevious attempt failed with: {error_context}\nPlease revise the reasoning."

                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        visited_node_ids=visited_node_ids,
                        informal_proof=str(e),
                        pseudocode="",
                        formal_proof="",
                        passed=False
                    )

        # If we get here, all depths and attempts failed
        return {
            'name': test_case.name,
            'passed': False,
            'visited_node_ids': visited_node_ids,
            'informal_proof': informal_proof,
            'pseudocode': None,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': self.max_attempts
        }
        
def load_test_data(file_path: str) -> List[TestCase]: 
    with open(file_path, 'r') as f: 
        data = [json.loads(line) for line in f]
        
    test_data = [
        TestCase(
            name=entry['name'],
            header=entry.get('header', ''),
            informal_prefix=entry.get('informal_prefix', ''),
            formal_statement=entry.get('formal_statement', ''),
            goal=entry.get('goal', '')
        )
        for entry in data if entry.get('split') == 'test'
    ]
    return test_data

def run_evaluation(prover: ThreeAgentProver, test_cases: List[TestCase], output_file: str): 
    results = [] 
    for test_case in test_cases: 
        result = prover.process_test_case(test_case) 
        results.append(result) 
        print(f"Processed {test_case.name}: {'Passed' if result['passed'] else 'Failed'}")
        
    num_passed = sum(1 for result in results if result['passed'])
    print(f"Total test cases: {len(results)}")
    print(f"Passed: {num_passed}")
    print(f"Failed: {len(results) - num_passed}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__": 
    load_dotenv()

# Initialize components
lean4_scheduler = Lean4ServerScheduler(
    max_concurrent_requests=3,
    timeout=300,
    memory_limit=10,
    name='verifier'
)

# Create each "agent"
proof_generator = ProofGenerator(model_type="gpt-4o-mini")
pseudocode_generator = PseudocodeGenerator(model_type="gpt-4o-mini")
auto_formalizer = AutoFormalizer()

try:
    # Load test data
    test_cases = load_test_data('datasets/minif2f.jsonl')
    print(f"Total test cases: {len(test_cases)}")
    
    # Create the three-agent prover
    prover = ThreeAgentProver(
        lean4_scheduler=lean4_scheduler,
        neo4j_uri=os.environ.get('NEO4J_URI'),
        neo4j_user=os.environ.get('NEO4J_USERNAME'),
        neo4j_password=os.environ.get('NEO4J_PASSWORD'),
        proof_generator=proof_generator,
        pseudocode_generator=pseudocode_generator,
        auto_formalizer=auto_formalizer,
        max_depth=2,
        max_attempts=1,
        log_file='RAG_three_agent_prover_results.csv'
    )
    
    # Run evaluation
    results = run_evaluation(
        prover,
        test_cases,
        'RAG_three_agent_prover_results.json'
    )
    
finally:
    lean4_scheduler.close()