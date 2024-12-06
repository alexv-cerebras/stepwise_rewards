from graphviz import Digraph
import textwrap
import uuid
import re

def visualize_tree(root_node, output_file="stepwise_tree"):
    """
    Create a visualization of the MCTS tree using graphviz with complete text display.
    
    Args:
        root_node (Node): Root node of the tree
        output_file (str): Name of the output file (without extension)
    """
    dot = Digraph(comment='Stepwise Tree')
    dot.attr(rankdir='TB')
    
    # Set default node attributes
    dot.attr('node', 
            shape='box', 
            style='rounded,filled', 
            fillcolor='white',
            fontname='Helvetica',
            margin='0.3,0.1')
    
    def escape_dot_string(s):
        """Escape special characters for DOT format"""
        if not isinstance(s, str):
            s = str(s)
        # Escape backslashes first, then quotes
        s = s.replace('\\', '\\\\').replace('"', '\\"')
        # Replace problematic characters
        s = s.replace('\n', '\\n')
        return s
    
    def wrap_text(text, width=60):
        """Wrap text to specific width while preserving structure"""
        if not isinstance(text, str):
            text = str(text)
        # Split on newlines first to preserve intentional line breaks
        lines = text.split('\n')
        wrapped_lines = []
        for line in lines:
            # Only wrap if line is longer than width
            if len(line) > width:
                wrapped = textwrap.wrap(line, width=width, 
                                      break_long_words=False, 
                                      replace_whitespace=False)
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append(line)
        return '\\n'.join(wrapped_lines)
    
    def add_node(node, parent_id=None):
        node_id = str(uuid.uuid4())
        
        # Build label with proper escaping
        label_parts = []
        
        # Add state with proper escaping and wrapping
        state_str = escape_dot_string(str(node.state))
        label_parts.append(f"State: {wrap_text(state_str)}")
        
        # Add metrics
        label_parts.append(f"Correct solutions: {node.correct_solutions}")
        label_parts.append(f"Total solutions: {node.total_solutions}")
        label_parts.append(f"Prob: {getattr(node, 'probability', 0):.2f}")
        
        # Join all parts with proper line breaks
        label = '\\n\\n'.join(label_parts)
        
        # Create the node with escaped label
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
    
    dot.attr('graph', ratio='compress')
    dot.attr('node', fontsize='10')
    dot.attr('edge', fontsize='10')

    # Render with error handling
    try:
        dot.render(output_file, 
                  view=True, 
                  format='svg',
                  cleanup=True)  # cleanup=True removes the intermediate DOT file
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        # Optionally save the DOT source for debugging
        with open(f"{output_file}.dot", "w") as f:
            f.write(dot.source)
        raise

    return dot
