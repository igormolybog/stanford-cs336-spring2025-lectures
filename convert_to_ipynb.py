import ast
import json
import os
import argparse
import sys
import base64
import mimetypes

# Functions that contribute to Markdown content
CONTENT_FUNCTIONS = {
    'text', 'link', 'image', 'system_text', 
    'named_link', 'article_link', 'blog_link', 'x_link', 'youtube_link'
}

class LectureConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)
        # Map of function names to their nodes
        self.functions = {node.name: node for node in self.tree.body if isinstance(node, ast.FunctionDef)}
        self.cells = []
        self.current_markdown = []
        self.current_code = []
        self.references = self.load_references()

    def load_references(self):
        """Parse references.py AND the current tree to resolve object names to titles and URLs."""
        refs = {}
        
        # 1. Load from references.py
        refs_path = os.path.join(os.path.dirname(self.file_path), 'references.py')
        if os.path.exists(refs_path):
            try:
                with open(refs_path, 'r') as f:
                    tree = ast.parse(f.read())
                refs.update(self._scan_for_refs(tree))
            except Exception:
                pass
        
        # 2. Load from current tree (shadowing is fine)
        refs.update(self._scan_for_refs(self.tree))
        return refs

    def _scan_for_refs(self, tree):
        """Helper to scan a tree for Reference assignments."""
        refs = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                # Handle Reference(title=..., url=...) or arxiv_reference(url)
                ref_title = ""
                ref_url = ""
                if stmt.value.func.id == 'Reference':
                    ref_title = self.get_arg_value_simple(stmt.value, 0, 'title')
                    ref_url = self.get_arg_value_simple(stmt.value, 1, 'url')
                elif stmt.value.func.id == 'arxiv_reference':
                    ref_url = self.get_arg_value_simple(stmt.value, 0)
                    ref_title = "" # Fallback to name
                
                if ref_url:
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            refs[name] = (ref_title or name, ref_url)
        return refs

    def get_arg_value_simple(self, node, arg_index, kwarg_name=None):
        """Minimal version for load_references to avoid recursion."""
        if arg_index < len(node.args) and isinstance(node.args[arg_index], ast.Constant):
            return str(node.args[arg_index].value)
        if kwarg_name:
            for kw in node.keywords:
                if kw.arg == kwarg_name and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
        return ""

    def flush_markdown(self):
        if self.current_markdown:
            source = "\n\n".join(self.current_markdown) + "\n"
            self.cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [source]
            })
            self.current_markdown = []

    def flush_code(self):
        if self.current_code:
            source = "".join(self.current_code).rstrip() + "\n"
            self.cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [source]
            })
            self.current_code = []

    def get_arg_value(self, node, arg_index, kwarg_name=None, context_body=None):
        """Extract a value, resolving references if they are known."""
        arg = None
        if arg_index < len(node.args):
            arg = node.args[arg_index]
        elif kwarg_name:
            for kw in node.keywords:
                if kw.arg == kwarg_name:
                    arg = kw.value
                    break
        
        if arg is None: return ""

        if isinstance(arg, ast.Constant):
            return str(arg.value)
        elif isinstance(arg, ast.Name):
            if arg.id in self.references:
                return self.references[arg.id]
            # Try to find assignment in context_body
            if context_body:
                for stmt in reversed(context_body):
                    if stmt == node: continue
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == arg.id:
                        if isinstance(stmt.value, ast.Constant):
                            return str(stmt.value.value)
                        # Special case for run_policy_gradient return values
                        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'run_policy_gradient':
                             return arg.id # Keep name if it's a complex dynamic value
            return arg.id
        elif isinstance(arg, ast.JoinedStr):
            # Very basic f-string resolution: just take the constant parts and replace variables with {name}
            res = ""
            for part in arg.values:
                if isinstance(part, ast.Constant):
                    res += str(part.value)
                elif isinstance(part, ast.FormattedValue):
                    if isinstance(part.value, ast.Name):
                        res += "{" + part.value.id + "}"
                    else:
                        res += "{...}"
            return res
        return ""

    def format_content_call(self, node, context_body=None):
        func_name = node.func.id
        if func_name == 'text':
            val = self.get_arg_value(node, 0, context_body=context_body)
            if isinstance(val, tuple): return val[0]
            return val
        elif func_name == 'link':
            arg0 = self.get_arg_value(node, 0, 'title', context_body=context_body)
            if isinstance(arg0, tuple):
                title, url = arg0
            else:
                title = arg0
                url = self.get_arg_value(node, 1, 'url', context_body=context_body)
            
            if not isinstance(url, str) or not url:
                if title in self.references:
                    title, url = self.references[title]
            
            if not url: return title
            return f"[{title}]({url})"
        elif func_name == 'image':
            path = self.get_arg_value(node, 0, context_body=context_body)
            if isinstance(path, tuple): path = path[1]
            
            width = self.get_arg_value(node, 1, 'width', context_body=context_body)
            
            # Check if it's a local path or a remote URL
            if path.startswith(('http://', 'https://')):
                if width:
                    return f'<img src="{path}" width="{width}" />'
                return f"![image]({path})"
            
            # Local image embedding
            try:
                # Resolve path relative to the input file
                abs_path = os.path.join(os.path.dirname(os.path.abspath(self.file_path)), path)
                if os.path.exists(abs_path):
                    with open(abs_path, 'rb') as f:
                        data = base64.b64encode(f.read()).decode('utf-8')
                    mime, _ = mimetypes.guess_type(abs_path)
                    if not mime:
                        mime = 'image/png' # Fallback
                    
                    data_uri = f"data:{mime};base64,{data}"
                    if width:
                        return f'<img src="{data_uri}" width="{width}" />'
                    return f"![image]({data_uri})"
            except Exception as e:
                print(f"Warning: Could not embed image {path}: {e}", file=sys.stderr)
            
            # Fallback to standard Markdown if anything fails
            if width:
                return f'<img src="{path}" width="{width}" />'
            return f"![image]({path})"
        elif func_name in ('article_link', 'blog_link', 'x_link', 'youtube_link'):
            url = self.get_arg_value(node, 0, context_body=context_body)
            if isinstance(url, tuple): url = url[1]
            return f"[link]({url})"
        elif func_name == 'named_link':
            name = self.get_arg_value(node, 0, context_body=context_body)
            url = self.get_arg_value(node, 1, context_body=context_body)
            if isinstance(name, tuple): name = name[0]
            if isinstance(url, tuple): url = url[1]
            return f"[{name}]({url})"
        return ""

    def process_body(self, body):
        for stmt in body:
            handled = False
            if isinstance(stmt, ast.Expr):
                expr_value = stmt.value
                if isinstance(expr_value, ast.Call) and isinstance(expr_value.func, ast.Name):
                    func_name = expr_value.func.id
                    if func_name in CONTENT_FUNCTIONS:
                        self.flush_code()
                        content = self.format_content_call(expr_value, context_body=body)
                        if content:
                            self.current_markdown.append(content)
                        handled = True
                    elif func_name in self.functions:
                        self.flush_code()
                        self.flush_markdown()
                        self.process_body(self.functions[func_name].body)
                        handled = True
                elif isinstance(expr_value, ast.Tuple):
                    all_content = True
                    line_markdown = []
                    for el in expr_value.elts:
                        if isinstance(el, ast.Call) and isinstance(el.func, ast.Name) and el.func.id in CONTENT_FUNCTIONS:
                            content = self.format_content_call(el, context_body=body)
                            if content:
                                line_markdown.append(content.strip())
                        else:
                            all_content = False
                            break
                    if all_content:
                        self.flush_code()
                        self.current_markdown.append(" ".join(filter(None, line_markdown)))
                        handled = True

            if not handled:
                self.flush_markdown()
                stmt_source = ast.get_source_segment(self.source, stmt)
                if stmt_source:
                    self.current_code.append(stmt_source + "\n")

    def convert(self, output_path):
        # 1. Capture top-level setup (imports, symbols, etc.)
        top_level_code = []
        for stmt in self.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                if stmt.value.func.id in ('Reference', 'arxiv_reference'):
                    continue
            if isinstance(stmt, ast.If):
                # Skip if __name__ == "__main__"
                if isinstance(stmt.test, ast.Compare) and \
                   isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "__name__":
                    continue
            
            stmt_source = ast.get_source_segment(self.source, stmt)
            if stmt_source:
                top_level_code.append(stmt_source + "\n")
        
        if top_level_code:
            self.current_code.extend(top_level_code)

        entry_point = 'main'
        if 'main' not in self.functions:
            for name in self.functions:
                if name.startswith('lecture_'):
                    entry_point = name
                    break
        
        if entry_point in self.functions:
            self.process_body(self.functions[entry_point].body)
        
        self.flush_code()
        self.flush_markdown()

        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        with open(output_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully converted {self.file_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lecture .py to .ipynb")
    parser.add_argument("-i", "--input", required=True, help="Input .py file")
    parser.add_argument("-o", "--output", help="Output .ipynb file")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.replace(".py", ".ipynb")
    
    converter = LectureConverter(input_path)
    converter.convert(output_path)
