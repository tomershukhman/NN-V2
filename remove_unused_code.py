#!/usr/bin/env python3
import argparse
import subprocess
import re
import ast
import os
import sys

def run_vulture():
    """
    Run Vulture on the current directory and capture its text output.
    Note: Vulture returns a nonzero exit code if unused code is detected.
    """
    try:
        # Run Vulture on the current directory.
        result = subprocess.run(
            ["vulture", "./dog_detector"],
            capture_output=True,
            text=True,
            check=False  # Do not raise an exception if vulture exits nonzero.
        )
    except Exception as e:
        print("Error running Vulture:", e)
        sys.exit(1)
        
    if result.stderr:
        # You might see warnings or info in stderr.
        print("Vulture stderr output:", result.stderr)
    return result.stdout.strip()

def parse_vulture_output(output):
    """
    Parse Vulture's output to extract unused function, method, and class definitions.
    Expected output format (one per line):
      filename:lineno: ... unused (function|method|class) 'name'
    This function skips any method whose name contains "forward" (case insensitive).
    """
    pattern = re.compile(
        r"^(?P<filename>.*?):(?P<lineno>\d+):.*?unused (function|method|class) '(?P<name>.*?)'"
    )
    matches = []
    for line in output.splitlines():
        m = pattern.match(line)
        if m:
            unused_type = m.group(3)
            name = m.group("name")
            # Skip any method that contains "forward" in its name.
            if unused_type == "method" and "forward" in name.lower():
                continue
            matches.append({
                "filename": m.group("filename"),
                "lineno": int(m.group("lineno")),
                "name": name,
                "type": unused_type
            })
    return matches

def get_ast_tree_and_lines(filename):
    """Parse the source file into an AST and return the AST and its source lines."""
    with open(filename, "r", encoding="utf8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        print(f"Syntax error in {filename}: {e}")
        return None, None
    return tree, source.splitlines()

def find_definition_node(tree, name, lineno):
    """
    Locate a function, method, or class definition in the AST that starts at the given line.
    Requires Python 3.8+ (for end_lineno support).
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name and hasattr(node, "lineno") and node.lineno == lineno:
                if hasattr(node, "end_lineno"):
                    return node
    return None

def remove_lines_from_source(lines, start, end):
    """
    Remove lines from start to end (inclusive) from a list of source lines.
    Parameters:
      - lines: list of source lines.
      - start: starting line number (1-indexed).
      - end: ending line number (1-indexed, inclusive).
    Returns a new list of lines.
    """
    return lines[:start - 1] + lines[end:]

def main():
    parser = argparse.ArgumentParser(
        description="Use Vulture to detect and optionally remove unused code (functions, methods, classes), ignoring any method containing 'forward'."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the detected unused code from the source files."
    )
    args = parser.parse_args()

    # Run Vulture and capture its output.
    vulture_output = run_vulture()
    vulture_results = parse_vulture_output(vulture_output)

    # Group results by file.
    files_to_remove = {}
    for item in vulture_results:
        filename = item.get("filename")
        lineno = item.get("lineno")
        name = item.get("name")
        if filename not in files_to_remove:
            files_to_remove[filename] = []
        files_to_remove[filename].append((lineno, name))

    if not files_to_remove:
        print("No unused code detected by Vulture.")
        return

    if not args.apply:
        print("Unused code detected (dry run):")
        for filename, entries in files_to_remove.items():
            for lineno, name in entries:
                print(f"  {filename}:{lineno} - {name}")
        print("\nRun with --apply to remove the detected code.")
        return

    # Apply changes: process each file and remove unused definitions.
    for filename, entries in files_to_remove.items():
        if not os.path.exists(filename):
            continue

        tree, lines = get_ast_tree_and_lines(filename)
        if tree is None or lines is None:
            continue

        nodes_to_remove = []
        for lineno, name in entries:
            node = find_definition_node(tree, name, lineno)
            if node is not None:
                nodes_to_remove.append((node.lineno, node.end_lineno, name))
            else:
                print(f"Warning: Could not locate AST node for {name} at line {lineno} in {filename}")

        # Sort removals in descending order to prevent shifting line numbers.
        nodes_to_remove.sort(key=lambda x: x[0], reverse=True)
        for start, end, name in nodes_to_remove:
            print(f"Removing {name} from {filename} (lines {start}-{end})")
            lines = remove_lines_from_source(lines, start, end)

        # Write the modified file back.
        with open(filename, "w", encoding="utf8") as f:
            f.write("\n".join(lines) + "\n")

    print("Unused code removed successfully.")

if __name__ == "__main__":
    main()
