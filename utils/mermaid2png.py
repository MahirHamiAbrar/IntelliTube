import subprocess
from pathlib import Path

def draw_png_dark(graph, output_path="graph.png", width=1200, height=800, background="transparent"):
    """
    Saves a PNG of the LangGraph with dark Mermaid styling using mmdc.

    Parameters:
        graph: Compiled LangGraph object (i.e. from graph.compile())
        output_path (str): Path to save the PNG
        width (int): Width of the PNG
        height (int): Height of the PNG
        background (str): Background style (e.g. 'transparent', '#1e1e2e')
    """
    mermaid_theme_header = '''%%{init: {
      "theme": "dark",
      "themeVariables": {
        "fontFamily": "Fira Code, monospace",
        "primaryColor": "#1e1e2e",
        "edgeLabelBackground": "#2e2e3e",
        "primaryTextColor": "#cdd6f4",
        "secondaryTextColor": "#89b4fa",
        "tertiaryColor": "#45475a",
        "nodeBorder": "#f38ba8",
        "lineColor": "#f38ba8"
      }
    }}%%\n'''

    # Get Mermaid graph content from LangGraph
    mermaid_code = mermaid_theme_header + graph.get_graph().draw_mermaid()

    # Write to temp .mmd file
    temp_mmd_path = Path(output_path).with_suffix(".mmd")
    temp_mmd_path.write_text(mermaid_code)

    # Run mmdc to generate PNG
    subprocess.run([
        "mmdc",
        "-i", str(temp_mmd_path),
        "-o", str(output_path),
        "-t", "dark",
        "-b", background,
        "-w", str(width),
        "-H", str(height)
    ], check=True)

    print(f"✅ PNG saved to: {output_path}")
