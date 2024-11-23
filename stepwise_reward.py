from binary_tree import BinaryTree, Node
from llm import openai_chat_completion
from collections import deque
import os
import opik
import re


os.environ["OPIK_PROJECT_NAME"] = "stepwise_rewards"
opik.configure(use_local=False)
    
model="llama3.1-8b"

stop_tokens = [
    "</s>",
    "<|eot_id|>",
    "Question:",
    "Question",
    "USER:",
    "USER",
    "ASSISTANT:",
    "ASSISTANT",
    "Instruction:",
    "Instruction",
    "Response:",
    "Response",
    "#",
    "# ",
    "###",
    "### "
]

with open("prompts/fewshot_prompt.txt") as f:
    examples = f.read()

with open("prompts/response_prompt.txt") as f:
    generation_prompt = f.read()


def concat_ost_steps(steps: list[str]) -> str:
    existing_ost_steps = "\n".join([f"Step {i}: {step}" for i, step in enumerate(steps, start=1)])
    next_step_id = len(steps) + 1
    return existing_ost_steps, next_step_id

def parse_step(step: str, step_idx: int):
    if step_idx == -1:
        # Pattern to match any step number followed by content
        step_pattern = r'Step \d+:(.*?)(?:Step \d+|$)'
        # Find all matches and take the last one
        matches = list(re.finditer(step_pattern, step, re.DOTALL))
        if matches:
            # Return content of the last step
            return matches[-1].group(1).strip()
    else:
        step_pattern = rf'Step {step_idx}:(.*?)(?:Step \d+|$)'
        match = re.search(step_pattern, step, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return step
    

class TreeStepwise(BinaryTree):
    def generate_children(self, parent_node: Node, finalize=False) -> list[Node]:
        reasoning_chain = parent_node.get_reasoning_chain()
        existing_ost_steps, next_step_id = concat_ost_steps(reasoning_chain)
        
        ost_input = (
            generation_prompt.format(instruction=self.problem, examples=examples)
            + existing_ost_steps
            + f"Step {next_step_id}:"
        )
        
        children = []
        n = 1 if finalize else self.max_children
        for _ in range(n):
            response = openai_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": ost_input,
                    },
                ],
                model=model,
                client=self.client,
                max_tokens=256,
                temperature=1.0
            )
            step_reasoning = response.choices[0].message.content
            assert step_reasoning is not None
            
            step_reasoning = parse_step(step_reasoning, next_step_id)
            node = Node(answer=step_reasoning, parent=parent_node)
            children.append(node)
        return children
                
    def finalize_answer(self):
        """
        Finalize answer in leaf nodes
        """
        candidates: list[Node] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            to_consider.extend(current_node.children)
            
            if not current_node.children and not current_node.is_leaf:
                candidates.append(current_node)
        
        for candidate in candidates:
            child = self.generate_children(candidate, finalize=True)
            candidate.add_child(child[0])
                
    def count_solutions(self, correct_answer: str):
        def func_to_check(answer, correct_answer):  
            answer = parse_step(answer, -1)
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given true answer and the answer to evaluate, return whether the answer is correct or not. "
                        "You should return correct if the answer is correct, and incorrect otherwise. Only return this without any additional information. "
                        "For the number comparison, consider a small margin of error. If one answer is 0.001 close to the other, consider it correct."
                    )
                },
                {
                    "role": "user",
                    "content": f"<true_answer>{correct_answer}<>\n<answer_to_evaluate>{answer}<>\n",
                }
            ] 
            for attempt in range(3):
                response = openai_chat_completion(
                    messages=messages,
                    model=model,
                    client=self.client,
                    max_tokens=64,
                    temperature=0.1
                )
                assert response.choices[0].message.content is not None
                
                answer = response.choices[0].message.content.lower().strip()
                if answer not in ['correct', 'incorrect']:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            },
                            {
                                "role": "user",
                                "content": "The response should be either correct or incorrect.",
                            },
                        ]
                    )
                    continue
                    
                if attempt == 2:
                    raise
                
                return answer == 'correct'
            
        self.root.count_solutions(func_to_check, correct_answer)
