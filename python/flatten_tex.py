#!/usr/bin/env python
r"""
Flatten a LaTeX document by recursively replacing \input and \include commands
with the actual file content, while removing figures, tables, and other non-text elements.
This creates a single file that's easier for LLMs to parse.
"""

import re
import os
import sys
from pathlib import Path

def remove_graphics_and_tables(content):
    """Remove or simplify graphics, tables, and other non-text elements."""

    # Remove \includegraphics commands
    content = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '[FIGURE REMOVED]', content)
    content = re.sub(r'\\includegraphics\{.*?\}', '[FIGURE REMOVED]', content)

    # Remove figure environments but keep captions
    def replace_figure(match):
        caption_match = re.search(r'\\caption\{(.*?)\}', match.group(0), re.DOTALL)
        label_match = re.search(r'\\label\{(.*?)\}', match.group(0))
        result = "\n[FIGURE REMOVED"
        if caption_match:
            result += f" - Caption: {caption_match.group(1)}"
        if label_match:
            result += f" - Label: {label_match.group(1)}"
        result += "]\n"
        return result

    content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', replace_figure, content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', replace_figure, content, flags=re.DOTALL)

    # Remove table environments but keep captions
    def replace_table(match):
        caption_match = re.search(r'\\caption\{(.*?)\}', match.group(0), re.DOTALL)
        label_match = re.search(r'\\label\{(.*?)\}', match.group(0))

        # Check if this table has an \input command (for generated tables)
        input_match = re.search(r'\\input\{(.*?)\}', match.group(0))

        result = "\n[TABLE REMOVED"
        if caption_match:
            result += f" - Caption: {caption_match.group(1)}"
        if label_match:
            result += f" - Label: {label_match.group(1)}"
        if input_match:
            result += f" - Source: {input_match.group(1)}"
        result += "]\n"
        return result

    content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', replace_table, content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{table\*\}.*?\\end\{table\*\}', replace_table, content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}', '[LONGTABLE REMOVED]', content, flags=re.DOTALL)

    # Remove lstlisting environments (code blocks)
    content = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '[CODE BLOCK REMOVED]', content, flags=re.DOTALL)

    # Remove graphicspath command
    content = re.sub(r'\\graphicspath\{.*?\}', '', content)

    return content

def flatten_tex(filepath, base_dir=None, processed_files=None, depth=0):
    r"""
    Recursively flatten a TeX file by replacing \input and \include commands.

    Args:
        filepath: Path to the TeX file to process
        base_dir: Base directory for relative paths
        processed_files: Set of already processed files to avoid cycles
        depth: Current recursion depth

    Returns:
        Flattened content as string
    """
    if processed_files is None:
        processed_files = set()

    if depth > 20:  # Prevent infinite recursion
        return f"% [ERROR: Maximum recursion depth exceeded]"

    filepath = Path(filepath)

    if base_dir is None:
        base_dir = filepath.parent
    else:
        base_dir = Path(base_dir)

    # Resolve the full path
    if not filepath.is_absolute():
        filepath = base_dir / filepath

    # Check if we've already processed this file
    abs_path = filepath.resolve()
    if abs_path in processed_files:
        return f"% [File already included: {filepath}]"

    processed_files.add(abs_path)

    # Read the file
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        # Try with .tex extension
        if not str(filepath).endswith('.tex'):
            return flatten_tex(str(filepath) + '.tex', base_dir, processed_files, depth)
        return f"% [File not found: {filepath}]"

    # Process \input and \include commands
    def replace_input(match):
        command = match.group(1)  # 'input' or 'include'
        filename = match.group(2).strip()

        # Skip table inputs (they're in the build directory)
        if 'build/tables' in filename or '../build/tables' in filename:
            return f"% [TABLE INPUT REMOVED: {filename}]"

        # Add comment to show what file is being included
        header = f"\n% ===== BEGIN INCLUDED FILE: {filename} =====\n"
        footer = f"\n% ===== END INCLUDED FILE: {filename} =====\n"

        # Get the directory of the current file for relative paths
        current_dir = filepath.parent

        # Recursively flatten the included file
        included_content = flatten_tex(filename, current_dir, processed_files, depth + 1)

        return header + included_content + footer

    # Replace \input{} and \include{} commands
    content = re.sub(r'\\(input|include)\{([^}]+)\}', replace_input, content)

    # Remove graphics and tables
    content = remove_graphics_and_tables(content)

    return content

def main():
    """Main function to flatten the paper."""
    # Get the report directory
    script_dir = Path(__file__).parent
    report_dir = script_dir.parent / 'report'
    main_tex = report_dir / 'paper.tex'

    # Output to build directory
    output_dir = script_dir.parent / 'build' / 'flattened'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'paper_flattened.tex'

    print(f"Flattening {main_tex}")
    print(f"Output: {output_file}")

    # Flatten the document
    flattened_content = flatten_tex(main_tex)

    # Add a header comment
    header = """% This is a flattened version of the paper for easier LLM parsing.
% All \\input and \\include commands have been recursively expanded.
% Figures, tables, and code blocks have been removed or simplified.
% Generated by python/flatten_tex.py
%
"""

    # Write the flattened content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(flattened_content)

    print(f"Successfully created flattened TeX file: {output_file}")

    # Also create a text-only version with minimal LaTeX markup
    text_output = output_dir / 'paper_text.txt'

    # Convert to more readable text format
    text_content = flattened_content

    # Remove LaTeX preamble
    preamble_end = text_content.find('\\begin{document}')
    if preamble_end > 0:
        text_content = text_content[preamble_end:]

    # Remove common LaTeX commands but keep structure
    text_content = re.sub(r'\\begin\{document\}', '', text_content)
    text_content = re.sub(r'\\end\{document\}', '', text_content)
    text_content = re.sub(r'\\maketitle', '', text_content)
    text_content = re.sub(r'\\tableofcontents', '', text_content)
    text_content = re.sub(r'\\newpage', '\n\n', text_content)
    text_content = re.sub(r'\\FloatBarrier', '', text_content)

    # Convert sections to markdown-style headers
    text_content = re.sub(r'\\section\*?\{([^}]+)\}', r'\n\n# \1\n', text_content)
    text_content = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n\n## \1\n', text_content)
    text_content = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n\n### \1\n', text_content)
    text_content = re.sub(r'\\paragraph\*?\{([^}]+)\}', r'\n\n#### \1\n', text_content)

    # Remove or simplify common LaTeX commands
    text_content = re.sub(r'\\label\{[^}]+\}', '', text_content)
    text_content = re.sub(r'\\ref\{([^}]+)\}', r'[\1]', text_content)
    text_content = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', text_content)
    text_content = re.sub(r'\\textit\{([^}]+)\}', r'\1', text_content)
    text_content = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text_content)
    text_content = re.sub(r'\\emph\{([^}]+)\}', r'\1', text_content)
    text_content = re.sub(r'\\texttt\{([^}]+)\}', r'\1', text_content)
    text_content = re.sub(r'\\TODO\{([^}]+)\}', r'[TODO: \1]', text_content)
    text_content = re.sub(r'\\TODOITEM', r'[TODO]', text_content)

    # Handle lists
    text_content = re.sub(r'\\begin\{itemize\}', '', text_content)
    text_content = re.sub(r'\\end\{itemize\}', '', text_content)
    text_content = re.sub(r'\\begin\{enumerate\}', '', text_content)
    text_content = re.sub(r'\\end\{enumerate\}', '', text_content)
    text_content = re.sub(r'\\item', '\n• ', text_content)

    # Clean up math mode markers (keep the math for readability)
    text_content = re.sub(r'\$([^$]+)\$', r'\1', text_content)
    text_content = re.sub(r'\\begin\{equation\*?\}', '\n', text_content)
    text_content = re.sub(r'\\end\{equation\*?\}', '\n', text_content)
    text_content = re.sub(r'\\begin\{align\*?\}', '\n', text_content)
    text_content = re.sub(r'\\end\{align\*?\}', '\n', text_content)
    text_content = re.sub(r'\\\[', '\n', text_content)
    text_content = re.sub(r'\\\]', '\n', text_content)

    # Clean up extra whitespace
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
    text_content = re.sub(r'[ \t]+', ' ', text_content)

    # Write the text version
    with open(text_output, 'w', encoding='utf-8') as f:
        f.write("# Derivative Estimation Methods Paper - Text Version\n\n")
        f.write("This is a simplified text version of the paper for easier LLM parsing.\n")
        f.write("LaTeX commands have been removed or simplified.\n\n")
        f.write(text_content)

    print(f"Successfully created text version: {text_output}")

if __name__ == '__main__':
    main()