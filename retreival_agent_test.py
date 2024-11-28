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
        prompt = f"""Given the following mathematical problem and context, generate a clear and detailed informal proof in natural language.
        Do not attempt to formalize the proof - focus only on explaining the mathematical reasoning clearly.

        Context:
        {context}

        Problem:
        {problem}

        Provide your proof in the following format:

        # Informal Proof:
        [Your natural language proof here]
        """
        
        if self.use_anthropic:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a mathematics expert focused on generating clear informal proofs."),
                ("user", prompt)
            ])
            chain = prompt_template | self.llm | StrOutputParser()
            return chain

class AutoFormalizer:
    """Responsible for converting informal proofs to Lean 4 formal proofs"""
    def __init__(self, model_type="deepseek-math-7b", temperature=0):
        # Note: This is a placeholder - replace with actual DeepSeek integration
        # For now, using GPT-4 to simulate the autoformalization
        self.llm = ChatOpenAI(
            model_name="gpt-4",  # Replace with DeepSeek
            temperature=temperature,
            openai_api_key=os.environ.get('OPENAI_API_KEY')
        )
    
    def formalize_proof(self, header: str, informal_proof: str, goal: str) -> str:
        prompt = f"""Convert the following informal mathematical proof into a formal Lean 4 proof.
        
        Lean 4 Header:
        {header}

        Informal Proof:
        {informal_proof}

        Goal to Prove:
        {goal}

        Provide the formalized proof in Lean 4 syntax.
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in formalizing mathematical proofs in Lean 4."),
            ("user", prompt)
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({}).strip()

class TwoAgentProver:
    def __init__(
        self,
        lean4_scheduler,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        proof_generator: ProofGenerator,
        auto_formalizer: AutoFormalizer,
        max_depth=3,
        top_k=5,
        max_attempts=3,
        log_file=None
    ):
        self.lean4_scheduler = lean4_scheduler
        self.proof_generator = proof_generator
        self.auto_formalizer = auto_formalizer
        self.max_depth = max_depth
        self.max_attempts = max_attempts
        
        # Initialize Neo4j and RAG components
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.retriever = Neo4jVectorRetriever(driver=self.driver, top_k=top_k)
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'two_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'depth', 'attempt', 'informal_proof', 'formal_proof', 'passed'])
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        request_id_list = self.lean4_scheduler.submit_all_request([formal_proof])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        return (result['pass'] == True and result['complete'] == True), result

    def _log_attempt(self, prompt: str, depth: int, attempt: int, 
                    informal_proof: str, formal_proof: str, passed: bool):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            informal_escaped = informal_proof.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            writer.writerow([prompt, depth, attempt, informal_escaped, formal_escaped, passed])

    def _get_rag_context(self, query: str, depth: int) -> str:
        query_embedding = self.retriever.get_query_embedding(query)
        context = ""
        current_node_id = None
        
        for d in range(depth + 1):
            if d == 0:
                current_node_id, content = self.retriever.get_top_node(query_embedding)
                if current_node_id is None:
                    return ""
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
                    context += f"{node_id}:\n{content}\n\n"
                
                current_node_id = top_neighbors[0][1]['id']
        
        return context.strip()

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_test_case(self, test_case: TestCase) -> Dict:
        for depth in range(self.max_depth + 1):
            print(f"\nTrying at depth {depth}")
            
            # Get RAG context for current depth
            rag_context = self._get_rag_context(test_case.informal_prefix, depth)
            if not rag_context:
                print(f"No context found at depth {depth}, moving to next depth")
                continue
            
            # Generate informal proof using RAG context
            informal_proof = self.proof_generator.generate_proof(
                rag_context, 
                test_case.informal_prefix
            )
            
            # Try multiple formalization attempts
            for attempt in range(self.max_attempts):
                try:
                    print(f"Processing {test_case.name}, Depth {depth}, Attempt {attempt + 1}")
                    
                    # Formalize the proof
                    formal_proof = self.auto_formalizer.formalize_proof(
                        test_case.header,
                        informal_proof,
                        test_case.goal
                    )
                    
                    # Verify with Lean
                    passes, output = self._verify_lean_proof(formal_proof)
                    
                    # Log the attempt
                    self._log_attempt(
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        informal_proof=informal_proof,
                        formal_proof=formal_proof,
                        passed=passes
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'informal_proof': informal_proof,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1
                        }
                    
                    # If verification failed, add error context for next attempt
                    if isinstance(output, dict) and 'errors' in output:
                        error_context = "\n".join([error.get('data', '') for error in output['errors']])
                        informal_proof += f"\nPrevious attempt failed with: {error_context}\nPlease revise the proof."
                
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    self._log_attempt(
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        informal_proof=str(e),
                        formal_proof="",
                        passed=False
                    )

        # If we get here, all depths and attempts failed
        return {
            'name': test_case.name,
            'passed': False,
            'informal_proof': informal_proof,
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

def run_evaluation(prover: TwoAgentProver, test_cases: List[TestCase], output_file: str):
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
        max_concurrent_requests=1,
        timeout=300,
        memory_limit=10,
        name='verifier'
    )
    
    proof_generator = ProofGenerator(model_type="gpt-4")
    auto_formalizer = AutoFormalizer()
    
    try:
        # Load test data
        test_cases = load_test_data('datasets/minif2f.jsonl')
        print(f"Total test cases: {len(test_cases)}")
        
        # Create two-agent prover
        prover = TwoAgentProver(
            lean4_scheduler=lean4_scheduler,
            neo4j_uri=os.environ.get('NEO4J_URI'),
            neo4j_user=os.environ.get('NEO4J_USERNAME'),
            neo4j_password=os.environ.get('NEO4J_PASSWORD'),
            proof_generator=proof_generator,
            auto_formalizer=auto_formalizer,
            log_file='two_agent_prover_results.csv'
        )
        
        # Run evaluation
        results = run_evaluation(
            prover,
            test_cases,
            'two_agent_prover_results.json'
        )
        
    finally:
        lean4_scheduler.close()