import ast
from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parent.parent / "notebooks" / "run_app_in_colab.ipynb"


def test_launcher_cells_are_valid_python_and_have_the_upload_widget():
    nb = nbformat.read(NB, as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 3
    for src in code:
        ast.parse(src)
    assert "files.upload()" in code[0] and "app/streamlit_app.py" in code[0]
    assert "\"clone\"" in code[0] and "REPO_URL" in code[0]        # clones from GitHub first, upload is the fallback
    assert "requirements.txt" in code[1] and "trycloudflare" in code[2] and "_stcore/health" in code[2]
    assert "serve_kernel_port_as_window" in code[2]          # Colab-native link, no external DNS needed
