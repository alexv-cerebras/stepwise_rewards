from full_tree import Tree
from utils import visualize_mcts_tree
import os
import openai
import asyncio


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
client = openai.Client(api_key=API_KEY, base_url=BASE_URL)

question = "Let $(x,y)$ be an ordered pair of real numbers that satisfies the equation $x^2+y^2=14x+48y$. What is the maximum value of $y$?"
answer = "49"

async def main():
    mcts = Tree(
        system="", 
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        client=client,
        max_depth=6,
        n_children=3
    )
    root = await mcts.run_async(question)
    root.count_solutions(answer)
    visualize_mcts_tree(root, "imgs/big_tree")

if __name__ == '__main__':
    asyncio.run(main())
