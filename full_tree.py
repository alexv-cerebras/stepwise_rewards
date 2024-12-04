import math
import random
import logging
from typing import List, Dict, Any, Tuple
import re
import asyncio
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: str, parent: 'Node' = None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.correct_solutions = 0
        self.total_solutions = 0
        self.value = 0.0
        
    def count_solutions(self, correct_answer: str):
        # if probability is already calculated, return
        if hasattr(self, 'probability'):
            return
        
        if not self.children:
            self.total_solutions += 1
            is_correct = self.extract_answer()[0] == correct_answer
            if is_correct:
                self.correct_solutions += 1
            # calculate probability for this node
            self.probability = self.correct_solutions / self.total_solutions
        else:
            for child in self.children:
                child.count_solutions(correct_answer)
            self.correct_solutions = sum([child.correct_solutions for child in self.children])
            self.total_solutions = sum([child.total_solutions for child in self.children])
            # calculate probability for this node
            self.probability = self.correct_solutions / (self.total_solutions + 1e-5)
        
    def extract_answer(self) -> Tuple[str, float]:
        logger.debug(f"Extracting answer from state: {self.state}")
        patterns = [
            r"The answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"The final answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"Therefore, the answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"So, the answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"Thus, the answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
            r"In conclusion, the answer is\s+(-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.state)
            if match:
                answer = match.group(1)
                confidence = 1.0
                logger.debug(f"Answer found using pattern '{pattern}': {answer}")
                return answer, confidence
        
        # If no pattern is found, try to extract any number
        numbers = re.findall(r'-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+)?', self.state)
        if numbers:
            answer = numbers[-1]  # Take the last number found
            confidence = 0.5  # Lower confidence as it's not in the expected format
            logger.debug(f"No pattern found. Using last number as answer: {answer}")
            return answer, confidence
        
        logger.warning("No answer found in the state.")
        return "", 0.0
    
    def evaluate(self) -> float:        
        # Extract the final answer from the node's state
        _, confidence = self.extract_answer()
        return confidence
    
    def is_terminal(self):
        return self.evaluate() == 1


class Tree:
    def __init__(self, system: str, client, model: str, max_depth: int = 3, n_children: int = 3):
        self.client = client
        self.model_name = model
        self.original_question = None 
        self.system = system
        self.max_depth = max_depth
        self.n_children = n_children

    # async def generate_response_async(self, prompt: str, temperature = 0.2) -> str:
    #     return await asyncio.to_thread(self.generate_response, prompt, temperature)
    
    # def run(self, question: str) -> str:
    #     return asyncio.run(self.run_async(question))
    
    def run(self, question: str):
        self.original_question = question  
        root = Node(question)      
        prev_level = [root]
        
        for _ in tqdm(range(self.max_depth)):    
            next_level = []
            for current_node in prev_level:
                children = self.expand(current_node)
                
                for child in children:
                    if not child.is_terminal():
                        next_level.append(child)
                    
            prev_level = next_level
        return root
    
    def expand(self, node: Node) -> Node:        
        temperatures = [0.1] +  (self.n_children-1)*[1.0]
        for temp in temperatures:
            prompt = self.create_prompt(node.state)
            new_state = self.generate_response(prompt, temp)
            child_node = Node(new_state, node)
            node.children.append(child_node)
        return node.children

    def generate_response(self, prompt: str, temperature) -> str:
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant focused on solving mathematical problems. Stick to the given question and avoid introducing new scenarios."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        generated_response = response.choices[0].message.content.strip()
        logger.debug(f"Generated response: {generated_response}")
        return generated_response

    def create_prompt(self, state: str) -> str:
        question = self.original_question if hasattr(self, 'original_question') else "the original question"
        prompt = f"""Given the current state: {state}
Generate the next logical step in solving {question}.
Your response should be a single, clear thought that moves towards the solution.
If you can determine the final answer at this step, state it clearly."""
    
        prompt = prompt + "\n\nIf you determine the final answer, explicitly state 'The final answer is [your numeric answer]' at the end of your response."
        logger.debug(f"Created prompt: {prompt}")
        return prompt

    def create_discriminator_prompt(self, partial_trajectory: List[Node]) -> str:
        states = [node.state for node in partial_trajectory]
        partial_reasoning = " ".join(states)
        return f"Given the partial reasoning:\n{partial_reasoning}\nComplete the reasoning to solve the problem:"
