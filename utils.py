from graphviz import Digraph
from full_tree import Node
import textwrap
import uuid


def print_tree(node: None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)

def visualize_mcts_tree(root_node, output_file="mcts_tree"):
    """
    Create a visualization of the MCTS tree using graphviz with complete text display.
    
    Args:
        root_node (MCTSNode): Root node of the MCTS tree
        output_file (str): Name of the output file (without extension)
    """
    dot = Digraph(comment='MCTS Tree')
    dot.attr(rankdir='TB')
    
    # Set default node attributes
    dot.attr('node', 
            shape='box', 
            style='rounded,filled', 
            fillcolor='white',
            fontname='Helvetica',
            margin='0.3,0.1')
    
    def wrap_text(text, width=60):
        """Wrap text to specific width while preserving JSON structure"""
        return '\n'.join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))
    
    def add_node(node, parent_id=None):
        node_id = str(uuid.uuid4())

        label = ""
        label += f"State: {wrap_text(node.state)}\n\n"
        
        # Create formatted label with complete text
        label += f"""Correct solutions: {node.correct_solutions}\n\nTotal solutions: {node.total_solutions}\n\nProb: {getattr(node, 'probability', 0):.2f}"""
        
        # Add node to graph with wrapped text
        dot.node(node_id, label)
        
        # Connect to parent if exists
        if parent_id:
            dot.edge(parent_id, node_id)
        
        # Recursively add children
        for child in node.children:
            add_node(child, node_id)
    
    # Start with root node
    add_node(root_node)
    
    # Set graph attributes for better layout
    dot.attr(rankdir='TB',
            ranksep='1.0',
            nodesep='0.8',
            pad='0.5')
    
    # Save the visualization with higher DPI and larger size
    dot.render(output_file, 
              view=True, 
              format='png')