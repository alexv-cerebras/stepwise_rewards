from __future__ import annotations
from pydantic import BaseModel
from opik import track
import tqdm


class Node:
    def __init__(self, question: str = "", answer: str = "", parent: Node | None = None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.correct_solutions = 0
        self.total_solutions = 0
        self.is_leaf = False

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def __repr__(self):
        node_args = []
        if self.question:
            node_args.append(f"question={self.question}")
        if self.answer:
            node_args.append(f"answer={self.answer}")
        if hasattr(self, 'probability'):
            node_args.append(f"prob={self.probability}")
            
        return f"Node({', '.join(node_args)})"
        
    def get_reasoning_chain(self):
        reasoning_chain = []
        current_node = self
        while current_node:
            if current_node.answer:
                reasoning_chain.append(current_node.answer)
            current_node = current_node.parent
        return reasoning_chain[::-1]
    
    def count_solutions(self, func_to_check: callable, correct_answer: str):
        # if probability is already calculated, return
        if hasattr(self, 'probability'):
            return
        
        if self.is_leaf:
            self.total_solutions += 1
            is_correct = func_to_check(self.answer, correct_answer)
            if is_correct:
                self.correct_solutions += 1
            # calculate probability for this node
            self.probability = self.correct_solutions / self.total_solutions
        else:
            for child in self.children:
                child.count_solutions(func_to_check, correct_answer)
            self.correct_solutions = sum([child.correct_solutions for child in self.children])
            self.total_solutions = sum([child.total_solutions for child in self.children])
            # calculate probability for this node
            self.probability = self.correct_solutions / (self.total_solutions + 1e-5)


class BinaryTree(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    client: any
    problem: str
    max_depth: int = 3
    root: Node = Node()
    max_children: int = 2

    # Logs
    critiques: list[str] = []
    selected_nodes: list[Node] = []

    def generate_children(self, parent_node: Node) -> list[Node]:
        """Generate children for a given parent node."""
        raise NotImplementedError
    
    def finalize_answer(self):
        """Finalize answer in leaf nodes."""
        raise NotImplementedError

    def initialize(self):
        """Generate a zero-shot answer."""
        self.root = Node(question=self.problem)

    @track(tags=["balanced_tree"])
    def run(self):        
        self.initialize()
        prev_level = [self.root]
        
        for _ in tqdm.tqdm(range(self.max_depth)):    
            next_level = []
            for current_node in prev_level:
                children = self.generate_children(current_node)
                
                for child in children:
                    current_node.add_child(child)
                    if not child.is_leaf:
                        next_level.append(child)
                    
            prev_level = next_level
            
        self.finalize_answer()
