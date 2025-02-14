import os
import json
import re
import csv
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Removed graph retrieval imports
# from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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

@dataclass
class ProofCandidate:
    proof_text: str
    score: float = 0.0
    metadata: Dict = None

class VotingProofGenerator:
    """Enhanced proof generator that uses a Best-of-N voting system"""
    
    def __init__(self, 
                 model_type: str = "gpt-4",
                 temperature: float = 0.7,
                 n_candidates: int = 5,
                 ranker_model: str = "gpt-4",  # Using GPT-4 for more reliable ranking
                 ranker_temperature: float = 0.0):
        self.client = OpenAI()
        self.model_type = model_type
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.ranker_model = ranker_model
        self.ranker_temperature = ranker_temperature
        
        self.generation_llm = ChatOpenAI(
            model_name=model_type,
            temperature=temperature,
            n=n_candidates,
        )
        
        self.ranker_llm = ChatOpenAI(
            model_name=ranker_model,
            temperature=ranker_temperature
        )
        
        self.n_candidates = n_candidates

    def generate_proof_candidates(self, context: str, problem: str) -> List[ProofCandidate]:
        """Generate multiple candidate proofs using the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are a mathematics expert focused on generating clear informal proofs.

Generate a clear and detailed informal proof in natural language.
Pay special attention to:
- Similar theorem statements
- Related proof techniques
- Mathematical patterns
- Definitions and axioms used

Context:
{context}

Problem to Prove:
{problem}

Provide your proof in this format:

# Informal Proof:
[Your natural language proof here]"""
                    }
                ],
                temperature=self.temperature,
                n=self.n_candidates,
            )
            
            return [
                ProofCandidate(proof_text=choice.message.content.strip())
                for choice in response.choices
            ]
        except Exception as e:
            print(f"Error generating proofs: {e}")
            return [ProofCandidate(proof_text="Error generating proof.")]

    def rank_candidates(self, candidates: List[ProofCandidate], context: str, problem: str) -> List[ProofCandidate]:
        """Rank candidates using OpenAI's JSON response format."""
        formatted_candidates = "\n\n".join([
            f"Candidate {i}:\n{c.proof_text}" 
            for i, c in enumerate(candidates)
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=self.ranker_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior mathematician evaluating mathematical proofs."
                    },
                    {
                        "role": "user",
                        "content": f"""Evaluate these candidate proofs:

Problem: {problem}

Context: {context}

{formatted_candidates}"""
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "proof_evaluation_schema",
                        "schema": {      
                            "type": "object",
                            "properties": {
                                "evaluations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "candidate_index": {"type": "integer"},
                                            "score": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 10
                                            },
                                            "justification": {"type": "string"}
                                        },
                                        "required": ["candidate_index", "score", "justification"]
                                    }
                                }
                            },
                            "required": ["evaluations"]
                        }
                    }
                },
                temperature=self.ranker_temperature
            )
            
            evaluations = json.loads(response.choices[0].message.content)["evaluations"]
            
            for eval_data in evaluations:
                idx = eval_data["candidate_index"]
                if idx < len(candidates):
                    candidates[idx].score = eval_data["score"]
                    candidates[idx].metadata = {
                        "justification": eval_data["justification"]
                    }
            
            candidates.sort(key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            print(f"Error in ranking: {e}")
            for i, candidate in enumerate(candidates):
                candidate.score = float(self.n_candidates - i)
        
        return candidates

    def generate_best_proof(self, context: str, problem: str) -> Tuple[ProofCandidate, List[ProofCandidate]]:
        """Generate and rank multiple proofs, returning the best candidate and all ranked candidates."""
        candidates = self.generate_proof_candidates(context, problem)
        ranked_candidates = self.rank_candidates(candidates, context, problem)
        return ranked_candidates[0], ranked_candidates

class TreeSearchProofGenerator(VotingProofGenerator):
    """Extends VotingProofGenerator with tree search capabilities"""
    
    def __init__(self,
                 max_depth: int = 3,
                 beam_width: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.beam_width = beam_width
        
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a mathematics expert refining proofs."),
            ("user", """Given a mathematical proof attempt, generate a refined version that addresses any gaps or weaknesses.

Original Proof:
{proof}

Problem:
{problem}

Context:
{context}

Previous Feedback (if any):
{feedback}

Generate a refined proof that improves upon the original.""")
        ])

    def _refine_proof(self, 
                      proof: str, 
                      context: str, 
                      problem: str,
                      feedback: str = "") -> List[ProofCandidate]:
        chain = self.refinement_prompt | self.generation_llm | StrOutputParser()
        
        responses = chain.batch([{
            "proof": proof,
            "problem": problem,
            "context": context,
            "feedback": feedback
        }] * self.n_candidates)
        
        return [ProofCandidate(proof_text=resp.strip()) for resp in responses]

    def tree_search(self, 
                    context: str, 
                    problem: str,
                    formal_verifier=None) -> Tuple[ProofCandidate, Dict]:
        """
        Perform tree search to find the best proof.
        
        Args:
            context: Mathematical context
            problem: Problem to prove
            formal_verifier: Optional function to verify proofs formally
            
        Returns:
            Tuple of (best candidate, search statistics)
        """
        stats = {
            "nodes_explored": 0,
            "formal_checks": 0,
            "successful_proofs": []
        }
        feedback = ""
        # Initial beam: using the direct context (no retrieval)
        current_beam = self.generate_proof_candidates(context, problem)
        current_beam = self.rank_candidates(current_beam, context, problem)
        current_beam = current_beam[:self.beam_width]
        
        for depth in range(self.max_depth):
            next_beam = []
            
            for candidate in current_beam:
                stats["nodes_explored"] += 1
                
                if formal_verifier:
                    stats["formal_checks"] += 1
                    try:
                        if formal_verifier(candidate.proof_text):
                            stats["successful_proofs"].append({
                                "depth": depth,
                                "proof": candidate
                            })
                            return candidate, stats
                    except Exception as e:
                        feedback = str(e)
                else:
                    feedback = ""
                
                refinements = self._refine_proof(
                    candidate.proof_text,
                    context,
                    problem,
                    feedback
                )
                next_beam.extend(refinements)
            
            if next_beam:
                next_beam = self.rank_candidates(next_beam, context, problem)
                current_beam = next_beam[:self.beam_width]
            else:
                break
        
        best_candidate = max(current_beam, key=lambda x: x.score)
        return best_candidate, stats

class AutoFormalizer:
    """Responsible for converting informal proofs to Lean 4 formal proofs"""
    def __init__(self, model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL", temperature=0.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = LLM(model=model_name, max_num_batched_tokens=8192, seed=1, trust_remote_code=True)
        self.temperature = temperature
    
    def formalize_proof(self, header: str, informal_proof: str, informal_prefix: str, formal_statement: str, goal: str) -> str:
        prompt = f"""You are a Lean 4 code generator. 
We have:
  HEADER:
{header}

  INFORMAL PROOF:
{informal_proof}

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
```  # End
"""
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048,
            top_p=0.1,
            n=1,
        )
        model_outputs = self.model.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )
        generated_text = model_outputs[0].outputs[0].text
        return generated_text.strip()

class TwoAgentProver:
    def __init__(
        self,
        lean4_scheduler,
        auto_formalizer: AutoFormalizer,
        max_depth=3,
        max_attempts=3,
        log_file=None,
        n_candidates=5,
        beam_width=3,
        search_depth=2,
        model_type="gpt-4o",
        ranker_model="gpt-4o"
    ):
        self.lean4_scheduler = lean4_scheduler
        self.proof_generator = TreeSearchProofGenerator(
            model_type=model_type,
            n_candidates=n_candidates,
            beam_width=beam_width,
            max_depth=search_depth,
            ranker_model=ranker_model
        )
        self.auto_formalizer = auto_formalizer
        self.max_depth = max_depth
        self.max_attempts = max_attempts
        
        # Graph retrieval components removed
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'two_agent_prover_log_{timestamp}.csv'
        self.log_file = log_file 
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'prompt', 'depth', 'attempt', 'visited_node_ids',
                'candidate_count', 'best_score', 'proof_scores',  
                'informal_proof', 'formal_proof', 'passed'
            ])
    
    def __del__(self):
        pass

    def _verify_lean_proof(self, formal_proof: str) -> Tuple[bool, Optional[Dict]]:
        request_id_list = self.lean4_scheduler.submit_all_request(
            [re.search(r'```lean4?\n(.*?)\n```', formal_proof, re.DOTALL).group(1)]
        )
        outputs_list = self.lean4_scheduler.get_all_request_outputs(request_id_list)
        result = outputs_list[0]
        print(result)
        return (result['pass'] == True and result['complete'] == True), result

    def _log_attempt(self, problem: str, prompt: str, depth: int, attempt: int, 
                     visited_node_ids: List[str], informal_proof: str, 
                     formal_proof: str, passed: bool,
                     candidate_count: int = 1, 
                     best_score: float = 0.0,   
                     proof_scores: List[float] = None):  
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            informal_escaped = informal_proof.replace('\n', '\\n')
            formal_escaped = formal_proof.replace('\n', '\\n')
            visited_str = ",".join(visited_node_ids)
            scores_str = ",".join(map(str, proof_scores)) if proof_scores else ""
            
            writer.writerow([
                problem, prompt, depth, attempt, visited_str,
                candidate_count, best_score, scores_str,
                informal_escaped, formal_escaped, passed
            ])

    def process_test_case(self, test_case: TestCase) -> Dict:
        # Use the test case's informal_prefix as the context (no graph retrieval)
        for depth in range(self.max_depth + 1):
            rag_context = test_case.informal_prefix
            visited_node_ids = []  # No nodes visited
            if not rag_context:
                continue
            
            def verify_proof(proof_text: str) -> bool:
                try:
                    formal_proof = self.auto_formalizer.formalize_proof(
                        test_case.header,
                        proof_text,
                        test_case.informal_prefix,
                        test_case.formal_statement,
                        test_case.goal
                    )
                    return self._verify_lean_proof(formal_proof)[0]
                except:
                    return False
            
            best_candidate, search_stats = self.proof_generator.tree_search(
                context=rag_context,
                problem=test_case.informal_prefix,
                formal_verifier=verify_proof
            )
            
            for attempt in range(self.max_attempts):
                try:
                    formal_proof = self.auto_formalizer.formalize_proof(
                        test_case.header,
                        best_candidate.proof_text,
                        test_case.informal_prefix,
                        test_case.formal_statement,
                        test_case.goal
                    )
                    
                    passes, output = self._verify_lean_proof(formal_proof)
                    
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        visited_node_ids=visited_node_ids,
                        informal_proof=best_candidate.proof_text,
                        formal_proof=formal_proof,
                        passed=passes,
                        candidate_count=search_stats["nodes_explored"],
                        best_score=best_candidate.score,
                        proof_scores=[p["proof"].score for p in search_stats.get("successful_proofs", [])]
                    )
                    
                    if passes:
                        print(f"Depth {depth}, Attempt {attempt + 1} succeeded")
                        return {
                            'name': test_case.name,
                            'passed': True,
                            'visited_node_ids': visited_node_ids,
                            'informal_proof': best_candidate.proof_text,
                            'lean_code': output.get("verified_code"),
                            'depth': depth,
                            'attempts': attempt + 1,
                            'search_stats': search_stats,
                            'best_score': best_candidate.score
                        }
                    
                    if isinstance(output, dict) and 'errors' in output:
                        error_context = "\n".join([error.get('data', '') for error in output['errors']])
                        best_candidate, new_stats = self.proof_generator.generate_best_proof(
                            rag_context,
                            test_case.informal_prefix + f"\nPrevious errors: {error_context}"
                        )
                        search_stats["nodes_explored"] += new_stats["nodes_explored"]
                
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    self._log_attempt(
                        problem=test_case.name,
                        prompt=test_case.informal_prefix,
                        depth=depth,
                        attempt=attempt + 1,
                        visited_node_ids=visited_node_ids,
                        informal_proof=str(e),
                        formal_proof="",
                        passed=False,
                        candidate_count=search_stats["nodes_explored"],
                        best_score=0.0,
                        proof_scores=[]
                    )

        return {
            'name': test_case.name,
            'passed': False,
            'visited_node_ids': visited_node_ids,
            'informal_proof': best_candidate.proof_text,
            'lean_code': None,
            'depth': self.max_depth,
            'attempts': self.max_attempts,
            'search_stats': search_stats,
            'best_score': best_candidate.score
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
    
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=3,
        timeout=300,
        memory_limit=10,
        name='verifier'
    )
    
    auto_formalizer = AutoFormalizer()
    
    try:
        test_cases = load_test_data('datasets/mustard_short.jsonl')
        print(f"Total test cases: {len(test_cases)}")
        
        # Create TwoAgentProver without any graph retrieval parameters
        prover = TwoAgentProver(
            lean4_scheduler=lean4_scheduler,
            auto_formalizer=auto_formalizer,
            model_type="o1-mini",              
            ranker_model="gpt-4o",   
            n_candidates=5,
            beam_width=3,
            search_depth=2,
            max_depth=2,
            max_attempts=1,
            log_file='ms_o1_tree.csv'
        )
        
        results = run_evaluation(
            prover,
            test_cases,
            'ms_o1_tree.json'
        )
        
    finally:
        lean4_scheduler.close()
