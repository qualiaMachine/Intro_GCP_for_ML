import os
import nbformat
import re
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# List of Markdown files to ignore (no conversion needed)
ignore_list = [
    "01-Introduction.md",
    "02-Notebooks-as-controllers.md",
]

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Regular expression to detect code blocks (matches ```language\n...\n```)
code_block_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)

# Languages that should stay as formatted markdown, not become code cells
SHELL_LANGS = {"sh", "bash", "shell", "zsh"}
# Languages that represent plain output/text — keep as markdown without any prefix
TEXT_LANGS = {"text", "output"}

def strip_solutions(md_content):
    """Remove Carpentries solution blocks so learners don't see answers in notebooks.

    If a solution contains a code block, replace it with an empty code cell
    placeholder so learners get a blank cell to work in.
    """
    def _replace_solution(match):
        body = match.group(0)
        if re.search(r'```\w*\n', body):
            return '```python\n# Your code here\n```'
        return ''

    return re.sub(
        r'^:{16,}\s*solution\s*\n.*?\n:{16,}\s*$',
        _replace_solution, md_content, flags=re.MULTILINE | re.DOTALL
    )

def strip_yaml_frontmatter(md_content):
    """Replace YAML front matter with a clean markdown title.

    Converts the full front matter block (title, teaching, exercises, etc.)
    into just a horizontal-rule-wrapped title so notebooks look clean.
    """
    m = re.match(r'^---\s*\n(.*?\n)---\s*\n', md_content, re.DOTALL)
    if not m:
        return md_content
    frontmatter = m.group(1)
    title_match = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', frontmatter, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        header = f"---\n{title}\n---"
    else:
        header = ""
    return header + "\n" + md_content[m.end():]


def strip_fenced_divs(text):
    """Remove Carpentries fenced-div markers (:::+ lines) from markdown."""
    return re.sub(r'^:{3,}.*$', '', text, flags=re.MULTILINE)

def split_markdown(md_content):
    """Splits Markdown content into separate Markdown and Code cells."""
    cells = []
    position = 0

    for match in code_block_pattern.finditer(md_content):
        # Extract text before the code block as Markdown
        before_code = md_content[position:match.start()].strip()
        if before_code:
            cells.append(new_markdown_cell(before_code))

        # Extract code block content
        lang = (match.group(1) or "").lower()
        code_content = match.group(2).strip()
        if code_content:
            if lang in SHELL_LANGS:
                # Keep shell blocks as readable markdown so learners
                # don't accidentally run them inside the notebook.
                md = f"**Run in Cloud Shell / terminal:**\n```{lang}\n{code_content}\n```"
                cells.append(new_markdown_cell(md))
            elif lang in TEXT_LANGS:
                # Plain text/output blocks stay as formatted markdown
                md = f"```{lang}\n{code_content}\n```"
                cells.append(new_markdown_cell(md))
            else:
                cells.append(new_code_cell(code_content))

        position = match.end()

    # Add any remaining Markdown content after the last code block
    remaining_md = md_content[position:].strip()
    if remaining_md:
        cells.append(new_markdown_cell(remaining_md))
    
    return cells

# Convert each Markdown file in episodes/
for filename in os.listdir(episodes_dir):
    if filename.endswith(".md") and filename not in ignore_list:
        md_path = os.path.join(episodes_dir, filename)
        ipynb_path = os.path.join(notebooks_dir, filename.replace(".md", ".ipynb"))

        # Read Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Clean up Carpentries formatting before converting
        md_content = strip_yaml_frontmatter(md_content)
        md_content = strip_solutions(md_content)
        md_content = strip_fenced_divs(md_content)
        notebook_cells = split_markdown(md_content)

        # Create Jupyter notebook
        nb = new_notebook(cells=notebook_cells)

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("Conversion complete! Excluded:", ", ".join(ignore_list))
