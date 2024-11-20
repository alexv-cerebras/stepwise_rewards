from mcts import MCTS, MCTSNode
from llm import openai_chat_completion
from collections import deque
import re

    
model="llama3.1-8b"
judge_model="llama3.1-8b"

critic_system_prompt=(
    "I will provide you with previous steps of reasoning. "
    "Provide a detailed and constructive critique to improve the answer for the current step only."
    "Highlight specific areas that need refinement or correction."
)
evaluate_system_prompt=(
    "I will provide you with previous steps of reasoning. "
    "Provide a reward score between -100 and 100 for the answer for the current step's quality, using very strict standards. "
    "Do not give a full score above 95. Make sure the reward score is an integer. "
    "Return *ONLY* the score."
)

answer_final_format = """<reasoning>Refined reasoning step goes here</reasoning>
<answer>Final answer</answer>
"""

answer_optional_format = """<reasoning>Refined reasoning step goes here</reasoning>
<answer>Final answer if you can answer based on the previous reasoning and current step. If you can't, just leave empty.</answer>
"""

refine_system_prompt="""# Instruction
I will provide you with previous steps of reasoning. Refine the answer for the current step only based on the critique and return only the refined reasoning step.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.

You should return in the following format:
{answer_format}
"""


def parse_step(step_text: str) -> tuple[str, str]:
    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', step_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', step_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    return reasoning, answer


class MCTSStepwise(MCTS):
    def self_refine(self, node: MCTSNode, answer_format: str = answer_optional_format) -> MCTSNode:
        reasoning_chain = node.get_reasoning_chain()
        reasoning_chain = '\n'.join(reasoning_chain[:-1])
        
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<reasoning_chain>\n{reasoning_chain}\n</reasoning_chain>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=judge_model,
            client=self.client,
            max_tokens=4000,
            temperature=0.1
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)
        
        refine_prompt = refine_system_prompt.format(answer_format=answer_format)
        
        messages=[
            {
                "role": "system",
                "content": refine_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<reasoning_chain>\n{reasoning_chain}\n</reasoning_chain>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                        f"<critique>\n{critique}\n</critique>",
                    ]
                ),
            },
        ]
        
        for attempt in range(3):
            refined_answer_response = openai_chat_completion(
                    messages=messages,
                    model=model,
                    client=self.client,
                    max_tokens=4000,
                    temperature=1.0,
                )
            assert refined_answer_response.choices[0].message.content is not None
            
            reasoning, answer = parse_step(refined_answer_response.choices[0].message.content)
            refined_answer = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
            
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": refined_answer_response.choices[0].message.content,
                    },
                    {
                        "role": "user",
                        "content": f"""Failed to parse response in the following format:
                        {answer_format}""",
                    },
                ]
            )
            
            if reasoning or attempt == 2:
                break   
        
        assert reasoning is not None 
        node = MCTSNode(answer=refined_answer, parent=node)
        if answer:
            node.is_leaf = True
        
        self.refinements.append(refined_answer)
        return node

    def _evaluate_answer(self, node: MCTSNode, temperature=0.8) -> int:
        reasoning_chain = node.get_reasoning_chain()
        reasoning_chain = '\n'.join(reasoning_chain[:-1])
        
        messages = [
            {
                "role": "system",
                "content": evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<reasoning_chain>\n{reasoning_chain}\n</reasoning_chain>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=judge_model,
                    client=self.client,
                    max_tokens=4000,
                    temperature=0.1
                )
                assert response.choices[0].message.content is not None
                
                answer = float(response.choices[0].message.content)
                return answer
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as float.",
                        },
                    ]
                )
                if attempt == 2:
                    raise
                
    def run(self):
        super().run()
        self.finalize_answer()
                
    def finalize_answer(self):
        """
        Finalize answer in leaf nodes
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            to_consider.extend(current_node.children)
            
            if not current_node.children and not current_node.is_leaf:
                candidates.append(current_node)
        
        for candidate in candidates:
            child = self.self_refine(candidate, answer_final_format)
            candidate.add_child(child)
                
    def count_solutions(self, correct_answer: str):
        def func_to_check(answer, correct_answer):  
            _, answer = parse_step(answer)
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
                    max_tokens=4000,
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
