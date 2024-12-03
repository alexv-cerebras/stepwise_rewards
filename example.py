from rstar import RStar
from utils import visualize_mcts_tree
import os
import openai

API_KEY = os.getenv("CEREBRAS_API_KEY")
BASE_URL = os.getenv("CEREBRAS_API_URL")
client = openai.Client(api_key=API_KEY, base_url=BASE_URL)

mcts = RStar(
    system="", 
    model="llama3.1-8b",
    client=client,
    max_depth=3
)

root, _, _ = mcts.solve("How many people live in Berlin?")
root.count_solutions("6,785,717")
visualize_mcts_tree(root, "imgs/rstar_v6")
