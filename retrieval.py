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

# Initialize Neo4j driver
uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
driver = GraphDatabase.driver(uri, auth=(user, password))

@dataclass
class TestCase:
    name: str
    header: str
    informal_prefix: str
    formal_statement: str
    goal: str

class BaseProver:
    def __init__(self, lean4_scheduler, model_name='gpt-4o', temperature=0):
        self.lean4_scheduler = lean4_scheduler
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name=model_name,
            temperature=temperature
        )
        self.answer_chain = self._create_answer_chain()
    
    def _create_answer_chain(self):
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
        return answer_prompt | self.llm | StrOutputParser()
    
    def generate_answer(self, context: str, question: str) -> str:
        response = self.answer_chain.invoke({"context": context, "question": question})
        return response.strip()
    
    def is_answer_acceptable(self, answer: str) -> Tuple[bool, Optional[Dict]]:
        request_id_list = self.lean4_scheduler.submit_all_request(
            [re.search(r'```[\w]*\n(.*?)\n```', answer, re.DOTALL).group(1)]
        )
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        print(outputs_list[0])
        return (outputs_list[0]['pass'] == True and 
                outputs_list[0]['complete'] == True), outputs_list[0]

    def process_test_case(self, test_case: TestCase, max_attempts: int = 3) -> Dict:
        input_prompt = """
Given this problem statement: 
{informal_prefix}
Complete the following Lean 4 code:
{header}
{formal_statement}
to reach this goal: 
{goal}
""".format(
            informal_prefix=test_case.informal_prefix,
            header=test_case.header,
            formal_statement=test_case.formal_statement,
            goal=test_case.goal
        )

        attempt = 0
        passes = False
        output = ""
        context = ""  # Initialize empty context for error feedback

        while attempt < max_attempts and not passes:
            try:
                print(f"Processing {test_case.name}, Attempt {attempt + 1}")
                
                # Add error context if available from previous attempt
                if context:
                    full_prompt = f"{input_prompt}\n\nPrevious attempt:\n```lean4\n{output.get('verified_code', '')}\n```\nfailed with error:\n{context}\nPlease fix these issues."
                    print(full_prompt)
                else:
                    full_prompt = input_prompt
                    
                answer = self.generate_answer("", full_prompt)  # Empty context for baseline
                passes, output = self.is_answer_acceptable(answer)
                
                if passes:
                    print(f"Attempt {attempt + 1} succeeded for {test_case.name}.")
                else:
                    print(f"Attempt {attempt + 1} failed for {test_case.name}. Retrying...")
                    # Extract error messages for context in next attempt
                    if isinstance(output, dict) and 'errors' in output:
                        context = "\n".join([error.get('data', '') for error in output['errors']])
            except Exception as e:
                print(f"Error processing {test_case.name} on attempt {attempt + 1}: {e}")
            finally:
                attempt += 1

        return {
            'name': test_case.name,
            'passed': passes,
            'lean_code': output.get("verified_code") if isinstance(output, dict) else None
        }

class RAGProver(BaseProver):
    def __init__(self, lean4_scheduler, neo4j_uri, neo4j_user, neo4j_password, max_depth=3, top_k=5, log_file=None, **kwargs):
        super().__init__(lean4_scheduler, **kwargs)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.max_depth = max_depth
        self.top_k = top_k
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.retriever = Neo4jVectorRetriever(driver=self.driver, top_k=top_k)
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'rag_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        # Create CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'depth', 'attempt', 'answer', 'passed'])
        
        
    def _log_attempt(self, prompt: str, depth: int, attempt: int, answer: str, passed: bool):
        """Log an attempt to the CSV file."""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escape newlines in answer for CSV compatibility
            answer_escaped = answer.replace('\n', '\\n')
            writer.writerow([prompt, depth, attempt, answer_escaped, passed])
        
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def process_test_case(self, test_case: TestCase, max_attempts: int = 3) -> Dict:
        input_prompt = """
Given this problem statement: 
{informal_prefix}
Complete the following Lean 4 code:
{header}
{formal_statement}
to reach this goal: 
{goal}""".format(
            informal_prefix=test_case.informal_prefix,
            header=test_case.header,
            formal_statement=test_case.formal_statement,
            goal=test_case.goal
        )
        
        for depth in range(self.max_depth + 1):
            print(f"\nTrying at depth {depth}")
            
            # Get context for current depth
            rag_context = self._get_rag_context(input_prompt, depth)
            print(rag_context)
            if not rag_context:
                print(f"No context found at depth {depth}, moving to next depth")
                continue

            # Try multiple attempts at current depth
            attempt = 0
            error_context = ""  # Initialize empty error context
            
            while attempt < max_attempts:
                try:
                    print(f"Processing {test_case.name}, Depth {depth}, Attempt {attempt + 1}")
                    
                    # Combine RAG context with error feedback
                    if error_context:
                        full_context = f"{rag_context}\n\nPrevious attempt:\n```lean4\n{output.get('verified_code', '')}\n```\nfailed with error:\n{error_context}Please fix these issues."
                    else:
                        full_context = rag_context
                    
                    # Generate and verify answer
                    answer = self.generate_answer(full_context, input_prompt)
                    passes, output = self.is_answer_acceptable(answer)
                    
                    # Log the attempt
                    self._log_attempt(
                        prompt=input_prompt,
                        depth=depth,
                        attempt=attempt + 1,
                        answer=answer,
                        passed=passes
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded for {test_case.name}.")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1
                        }
                    else:
                        print(f"Depth {depth}, Attempt {attempt + 1} failed for {test_case.name}. Retrying...")
                        # Extract error messages for context in next attempt
                        if isinstance(output, dict) and 'errors' in output:
                            error_context = "\n".join([error.get('data', '') for error in output['errors']])
                
                except Exception as e:
                    print(f"Error processing {test_case.name} on attempt {attempt + 1}: {e}")
                    # Log failed attempts due to exceptions
                    self._log_attempt(
                        prompt=input_prompt,
                        depth=depth,
                        attempt=attempt + 1,
                        answer=str(e),
                        passed=False
                    )
                finally:
                    attempt += 1
        # Log final failed state if we exhaust all depths and attempts
        self._log_attempt(
            prompt=input_prompt,
            depth=self.max_depth,
            attempt=max_attempts,
            answer="Exhausted all attempts",
            passed=False
        )

        # If we get here, all depths and attempts failed
        return {
            'name': test_case.name,
            'passed': False,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': max_attempts
        }

    def _get_rag_context(self, query: str, target_depth: int) -> str:
        """Modified to only return context up to target_depth"""
        query_embedding = self.retriever.get_query_embedding(query)
        context = ""
        current_node_id = None

        for depth in range(target_depth + 1):
            if depth == 0:
                current_node_id, content = self.retriever.get_top_node(query_embedding)
                if current_node_id is None:
                    return ""
                context += str(current_node_id) + ":\n" + content + "\n\n"
            else:
                neighbors = self.retriever.get_neighbors(current_node_id, self.top_k)
                if not neighbors:
                    break
                    
                # Sort neighbors by similarity
                neighbor_similarities = []
                for neighbor in neighbors:
                    neighbor_embedding = neighbor['embedding']
                    if neighbor_embedding is None:
                        continue
                    similarity = self._cosine_similarity(query_embedding, neighbor_embedding)
                    neighbor_similarities.append((similarity, neighbor))
                
                # Sort and take top k
                neighbor_similarities.sort(reverse=True, key=lambda x: x[0])
                top_neighbors = neighbor_similarities[:self.top_k]
                
                if not top_neighbors:
                    break
                    
                # Add content from all top k neighbors
                for similarity, neighbor in top_neighbors:
                    content = neighbor.get('content', '')
                    node_id = neighbor.get('id')
                    context += str(node_id) + ":\n" + content + "\n\n"
                
                # Use most similar for next iteration
                best_similarity, best_neighbor = top_neighbors[0]
                current_node_id = best_neighbor['id']

        return context.strip()
    
    def _cosine_similarity(self, a, b):
        """Helper method to compute cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class AnthropicProver:
    def __init__(self, lean4_scheduler, model_name='claude-3-5-sonnet-latest', temperature=0):
        self.lean4_scheduler = lean4_scheduler
        self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        self.model_name = model_name
        self.temperature = temperature
        
    def generate_answer(self, context: str, question: str) -> str:
        system_prompt = """You are a math expert. Now please come up with a math problem according to the following requirements. 
        The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language 
        (indicated by ``Informal proof:''), and a translated formal solution in Lean 4 (indicated by ``Formal proof in Lean 4:''). 
        Please note that the informal proof and the formal proof need to be identical."""
        
        message = f"""Context:
{context}

You must respond in the following format: 

# Problem: 
{question}

# Informal proof: ...

# Formal proof in Lean 4: 
```lean4 
...
```
"""
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        
        return response.content[0].text.strip()
    
    def is_answer_acceptable(self, answer: str) -> Tuple[bool, Optional[Dict]]:
        # Extract Lean4 code from the answer using regex
        lean_code_match = re.search(r'```[\w]*\n(.*?)\n```', answer, re.DOTALL)
        if not lean_code_match:
            return False, {"errors": [{"data": "No Lean4 code found in response"}]}
        
        lean_code = lean_code_match.group(1)
        
        # Submit code to Lean4 scheduler for verification
        request_id_list = self.lean4_scheduler.submit_all_request([lean_code])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        print(outputs_list[0])
        
        return (outputs_list[0]['pass'] == True and 
                outputs_list[0]['complete'] == True), outputs_list[0]

    def process_test_case(self, test_case: TestCase, max_attempts: int = 3) -> Dict:
        input_prompt = """
Given this problem statement: 
{informal_prefix}
Complete the following Lean 4 code:
{header}
{formal_statement}
to reach this goal: 
{goal}
""".format(
            informal_prefix=test_case.informal_prefix,
            header=test_case.header,
            formal_statement=test_case.formal_statement,
            goal=test_case.goal
        )

        attempt = 0
        passes = False
        output = ""
        context = ""  # Initialize empty context for error feedback

        while attempt < max_attempts and not passes:
            try:
                print(f"Processing {test_case.name}, Attempt {attempt + 1}")
                
                # Add error context if available from previous attempt
                if context:
                    full_prompt = f"{input_prompt}\n\nPrevious attempt:\n```lean4\n{output.get('verified_code', '')}\n```\nfailed with error:\n{context}\nPlease fix these issues."
                    print(full_prompt)
                else:
                    full_prompt = input_prompt
                    
                answer = self.generate_answer("", full_prompt)  # Empty context for baseline
                passes, output = self.is_answer_acceptable(answer)
                
                if passes:
                    print(f"Attempt {attempt + 1} succeeded for {test_case.name}.")
                else:
                    print(f"Attempt {attempt + 1} failed for {test_case.name}. Retrying...")
                    # Extract error messages for context in next attempt
                    if isinstance(output, dict) and 'errors' in output:
                        context = "\n".join([error.get('data', '') for error in output['errors']])
            except Exception as e:
                print(f"Error processing {test_case.name} on attempt {attempt + 1}: {e}")
            finally:
                attempt += 1

        return {
            'name': test_case.name,
            'passed': passes,
            'lean_code': output.get("verified_code") if isinstance(output, dict) else None
        }

class AnthropicRAGProver:
    def __init__(
        self, 
        lean4_scheduler, 
        neo4j_uri, 
        neo4j_user, 
        neo4j_password, 
        max_depth=3, 
        top_k=5, 
        log_file=None, 
        model_name='claude-3-5-sonnet-latest', 
        temperature=0
    ):
        # Initialize Anthropic client
        self.lean4_scheduler = lean4_scheduler
        self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Neo4j and RAG components
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.max_depth = max_depth
        self.top_k = top_k
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.retriever = Neo4jVectorRetriever(driver=self.driver, top_k=top_k)
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'anthropic_rag_prover_log_{timestamp}.csv'
        self.log_file = log_file
        
        # Create CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'depth', 'attempt', 'answer', 'passed'])
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def generate_answer(self, context: str, question: str) -> str:
        system_prompt = """You are a math expert. Now please come up with a math problem according to the following requirements. 
        The math problem should contain a question part (indicated by ``Problem: ''), a corresponding solution in natural language 
        (indicated by ``Informal proof:''), and a translated formal solution in Lean 4 (indicated by ``Formal proof in Lean 4:''). 
        Please note that the informal proof and the formal proof need to be identical."""
        
        message = f"""Context:
{context}

You must respond in the following format: 

# Problem: 
{question}

# Informal proof: ...

# Formal proof in Lean 4: 
```lean4 
...
```
"""
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        
        return response.content[0].text.strip()

    def is_answer_acceptable(self, answer: str) -> Tuple[bool, Optional[Dict]]:
        lean_code_match = re.search(r'```[\w]*\n(.*?)\n```', answer, re.DOTALL)
        if not lean_code_match:
            return False, {"errors": [{"data": "No Lean4 code found in response"}]}
        
        lean_code = lean_code_match.group(1)
        request_id_list = self.lean4_scheduler.submit_all_request([lean_code])
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        print(outputs_list[0])
        
        return (outputs_list[0]['pass'] == True and 
                outputs_list[0]['complete'] == True), outputs_list[0]

    def _log_attempt(self, prompt: str, depth: int, attempt: int, answer: str, passed: bool):
        """Log an attempt to the CSV file."""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escape newlines in answer for CSV compatibility
            answer_escaped = answer.replace('\n', '\\n')
            writer.writerow([prompt, depth, attempt, answer_escaped, passed])

    def _get_rag_context(self, query: str, target_depth: int) -> str:
        """Get RAG context up to target_depth"""
        query_embedding = self.retriever.get_query_embedding(query)
        context = ""
        current_node_id = None

        for depth in range(target_depth + 1):
            if depth == 0:
                current_node_id, content = self.retriever.get_top_node(query_embedding)
                if current_node_id is None:
                    return ""
                context += str(current_node_id) + ":\n" + content + "\n\n"
            else:
                neighbors = self.retriever.get_neighbors(current_node_id, self.top_k)
                if not neighbors:
                    break
                    
                # Sort neighbors by similarity
                neighbor_similarities = []
                for neighbor in neighbors:
                    neighbor_embedding = neighbor['embedding']
                    if neighbor_embedding is None:
                        continue
                    similarity = self._cosine_similarity(query_embedding, neighbor_embedding)
                    neighbor_similarities.append((similarity, neighbor))
                
                # Sort and take top k
                neighbor_similarities.sort(reverse=True, key=lambda x: x[0])
                top_neighbors = neighbor_similarities[:self.top_k]
                
                if not top_neighbors:
                    break
                    
                # Add content from all top k neighbors
                for similarity, neighbor in top_neighbors:
                    content = neighbor.get('content', '')
                    node_id = neighbor.get('id')
                    context += str(node_id) + ":\n" + content + "\n\n"
                
                # Use most similar for next iteration
                best_similarity, best_neighbor = top_neighbors[0]
                current_node_id = best_neighbor['id']

        return context.strip()
    
    def _cosine_similarity(self, a, b):
        """Helper method to compute cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_test_case(self, test_case: TestCase, max_attempts: int = 3) -> Dict:
        input_prompt = """
Given this problem statement: 
{informal_prefix}
Complete the following Lean 4 code:
{header}
{formal_statement}
to reach this goal: 
{goal}""".format(
            informal_prefix=test_case.informal_prefix,
            header=test_case.header,
            formal_statement=test_case.formal_statement,
            goal=test_case.goal
        )
        
        for depth in range(self.max_depth + 1):
            print(f"\nTrying at depth {depth}")
            
            # Get context for current depth
            rag_context = self._get_rag_context(input_prompt, depth)
            print(rag_context)
            if not rag_context:
                print(f"No context found at depth {depth}, moving to next depth")
                continue

            # Try multiple attempts at current depth
            attempt = 0
            error_context = ""
            
            while attempt < max_attempts:
                try:
                    print(f"Processing {test_case.name}, Depth {depth}, Attempt {attempt + 1}")
                    
                    # Combine RAG context with error feedback
                    if error_context:
                        full_context = f"{rag_context}\n\nPrevious attempt:\n```lean4\n{output.get('verified_code', '')}\n```\nfailed with error:\n{error_context}Please fix these issues."
                    else:
                        full_context = rag_context
                    
                    # Generate and verify answer
                    answer = self.generate_answer(full_context, input_prompt)
                    passes, output = self.is_answer_acceptable(answer)
                    
                    # Log the attempt
                    self._log_attempt(
                        prompt=input_prompt,
                        depth=depth,
                        attempt=attempt + 1,
                        answer=answer,
                        passed=passes
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded for {test_case.name}.")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1
                        }
                    else:
                        print(f"Depth {depth}, Attempt {attempt + 1} failed for {test_case.name}. Retrying...")
                        # Extract error messages for context in next attempt
                        if isinstance(output, dict) and 'errors' in output:
                            error_context = "\n".join([error.get('data', '') for error in output['errors']])
                
                except Exception as e:
                    print(f"Error processing {test_case.name} on attempt {attempt + 1}: {e}")
                    self._log_attempt(
                        prompt=input_prompt,
                        depth=depth,
                        attempt=attempt + 1,
                        answer=str(e),
                        passed=False
                    )
                finally:
                    attempt += 1

        # Log final failed state if we exhaust all depths and attempts
        self._log_attempt(
            prompt=input_prompt,
            depth=self.max_depth,
            attempt=max_attempts,
            answer="Exhausted all attempts",
            passed=False
        )

        return {
            'name': test_case.name,
            'passed': False,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': max_attempts
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

def run_evaluation(prover, test_cases: List[TestCase], output_file: str):
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

# Usage example:
if __name__ == "__main__":
    load_dotenv()
    
    # Initialize resources
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=1,
        timeout=300,
        memory_limit=10,
        name='verifier'
    )

    try:
        # Load test data
        test_cases = load_test_data('datasets/minif2f.jsonl')
        print(f"Total test cases: {len(test_cases)}")
        print(test_cases[0])
        
        # Run baseline evaluation
        # baseline_prover = BaseProver(lean4_scheduler)
        # baseline_results = run_evaluation(
        #     baseline_prover, 
        #     test_cases, 
        #     '4o_minif2f_baseline_results.json'
        # )
        
        # Run Anthropic evaluation
        anthropic_prover = AnthropicProver(lean4_scheduler)
        anthropic_results = run_evaluation(
            anthropic_prover,
            test_cases,
            '3.5_sonnet_minif2f_results.json'
        )
        
        # Run RAG evaluation
        rag_prover = RAGProver(
            lean4_scheduler=lean4_scheduler,
            neo4j_uri=os.environ.get('NEO4J_URI'),
            neo4j_user=os.environ.get('NEO4J_USERNAME'),
            neo4j_password=os.environ.get('NEO4J_PASSWORD'),
            log_file='4o_mini_minif2f_rag_log.csv'
        )
        # rag_prover = AnthropicRAGProver(
        #     lean4_scheduler=lean4_scheduler,
        #     neo4j_uri=os.environ.get('NEO4J_URI'),
        #     neo4j_user=os.environ.get('NEO4J_USERNAME'),
        #     neo4j_password=os.environ.get('NEO4J_PASSWORD'),
        #     log_file='3.5_sonnet_rag_results.csv'
        # )
        # rag_results = run_evaluation(
        #     rag_prover,
        #     test_cases,
        #     '3.5_sonnet_minif2f_rag_results.json'
        # )
        
    finally:
        # Clean up resources
        lean4_scheduler.close()
