from __future__ import annotations
import random
import math
from collections import deque
from enum import Enum
from pydantic import BaseModel
import tqdm
import numpy as np
from pydantic import BaseModel


ROOT_UCT_SCORE = 10_000_000

class MCTSNode:
    def __init__(self, answer: str, parent: MCTSNode | None = None):
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.Q = 0
        self.reward_samples = []
        self.correct_solutions = 0
        self.total_solutions = 0
        self.is_leaf = False

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        if hasattr(self, 'probability'):
            return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits}, reward_samples={self.reward_samples}, prob={self.probability})"
        else:
            return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits}, reward_samples={self.reward_samples})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2
        
    def get_reasoning_chain(self):
        reasoning_chain = []
        current_node = self
        while current_node:
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


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3
    RANDOM = 4
    FULL_TREE = 5


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class MCTS(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    client: any
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.FULL_TREE
    # initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(answer="I don't know.")

    # Logs
    critiques: list[str] = []
    refinements: list[str] = []
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
        reward = self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children

    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            to_consider.extend(current_node.children)
            
            if self.selection_policy == SelectionPolicy.FULL_TREE:
                if not self.is_fully_expanded(current_node) and not current_node.is_leaf:
                    return current_node
            elif not self.is_fully_expanded(current_node) and not current_node.is_leaf:
                candidates.append(current_node)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.RANDOM:
            selected_pair_idx = random.choices(
                range(len(candidates)), k=1
            )[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weights=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        """Generate a zero-shot answer."""
        raise NotImplementedError()

    def initialize(self):
        """Generate a zero-shot answer."""
        self.root = MCTSNode(answer="I don't know.")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer
