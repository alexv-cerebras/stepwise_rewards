from full_tree import Tree
from utils import visualize_mcts_tree
import os
import openai

API_KEY = os.getenv("CEREBRAS_API_KEY")
BASE_URL = os.getenv("CEREBRAS_API_URL")
client = openai.Client(api_key=API_KEY, base_url=BASE_URL)

mcts = Tree(
    system="", 
    model="llama3.1-8b",
    client=client,
    max_depth=3,
    n_children=3
)

root = mcts.run("How many people in Berlin?")
root.count_solutions("6,785,717")
visualize_mcts_tree(root, "imgs/full_tree_v3")
