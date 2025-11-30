"""Display notebook cell outputs in a readable format."""

import json
import sys
from pathlib import Path


def display_notebook_outputs(notebook_path: Path):
    """Display all cell outputs from a notebook."""
    with open(notebook_path, encoding="utf-8") as f:
        notebook = json.load(f)

    print("=" * 80)
    print(f"NOTEBOOK: {notebook_path.name}")
    print("=" * 80)
    print()

    cell_num = 0
    for cell in notebook["cells"]:
        cell_num += 1
        cell_type = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown" and source.strip():
            print(f"\n{'='*80}")
            print(f"CELL {cell_num} - MARKDOWN")
            print("=" * 80)
            print(source)
        elif cell_type == "code":
            outputs = cell.get("outputs", [])
            if outputs or source.strip():
                print(f"\n{'='*80}")
                print(f"CELL {cell_num} - CODE")
                print("=" * 80)

                # Show code preview if non-trivial
                if source.strip() and len(source.split("\n")) <= 15:
                    print("Code:")
                    print("-" * 80)
                    print(source.rstrip())
                    print("-" * 80)
                elif source.strip():
                    lines = source.split("\n")
                    print(f"Code preview ({len(lines)} total lines):")
                    print("-" * 80)
                    print("\n".join(lines[:3]))
                    print("...")
                    print("-" * 80)

                # Show outputs
                if outputs:
                    print("\nOutput:")
                    print("-" * 80)
                    for output in outputs:
                        output_type = output.get("output_type", "")
                        if output_type == "stream":
                            text = "".join(output.get("text", []))
                            if text.strip():
                                print(text, end="")
                        elif output_type == "execute_result" or output_type == "display_data":
                            data = output.get("data", {})
                            if "text/plain" in data:
                                print("".join(data["text/plain"]), end="")
                            elif "text/html" in data:
                                # Show HTML as plain text indicator
                                html_text = "".join(data["text/html"])
                                print(f"[HTML output: {len(html_text)} characters]")
                        elif output_type == "error":
                            error_name = output.get("ename", "Error")
                            error_value = output.get("evalue", "")
                            traceback = output.get("traceback", [])
                            print(f"ERROR: {error_name}: {error_value}")
                            if traceback:
                                print("\n".join(traceback))
                    print("-" * 80)
                else:
                    print("\n[No output - cell has not been executed]")

    print("\n" + "=" * 80)
    print("END OF NOTEBOOK")
    print("=" * 80)


if __name__ == "__main__":
    notebook_path = Path("notebooks/phase4_api_serving.ipynb")
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])

    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)

    display_notebook_outputs(notebook_path)
