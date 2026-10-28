import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/01_data_preprocessing_eda.ipynb"
SOURCE_DIRECTORY = PROJECT_ROOT / "src/earnings_surprise"
OUTPUT_PATH = PROJECT_ROOT / "submission/EDA_Preprocessing.zip"

MAX_ARCHIVE_BYTES = 30_000_000

SUBMISSION_INSTRUCTIONS = """# EDA Notebook

### Requirements

- Python 3.12
- Internet connection for the dataset download

### Setup

From the extracted ZIP directory:

1. Create a virtual environment:

   python3.12 -m venv .venv

2. Activate the environment:

   source .venv/bin/activate

3. Install the project and dependencies:

   python -m pip install -e ".[dev]"

4. Open notebooks/01_data_preprocessing_eda.ipynb in JupyterLab or VS Code

5. Select the virtual environment as the notebook kernel

6. Restart the kernel and run all cells

The notebook downloads and verifies the pinned dataset automatically if it is not already present
"""


def validate_notebook() -> None:
    """checks that the notebook contains executed results"""
    with NOTEBOOK_PATH.open(encoding="utf-8") as file:
        notebook = json.load(file)

    code_cells = [
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    if not code_cells:
        raise ValueError("No code cells found in the notebook")

    for cell in code_cells:
        if cell.get("execution_count") is None:
            raise ValueError("Notebook contains unexecuted code cells")

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError("Notebook contains execution errors")


def build_submission() -> None:
    """packages the executed notebook and required Python files"""
    required_files = [
        NOTEBOOK_PATH,
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing file: {file_path}")

    if not SOURCE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Missing source directory: {SOURCE_DIRECTORY}"
        )

    validate_notebook()

    source_files = sorted(SOURCE_DIRECTORY.rglob("*.py"))

    if not source_files:
        raise ValueError("No Python source files found")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(OUTPUT_PATH, "w", ZIP_DEFLATED) as archive:
        for file_path in required_files + source_files:
            relative_path = file_path.relative_to(PROJECT_ROOT)
            archive.write(file_path, arcname=str(relative_path))

        archive.writestr(
            "RUN_EDA.md",
            SUBMISSION_INSTRUCTIONS,
        )

    archive_size = OUTPUT_PATH.stat().st_size

    if archive_size > MAX_ARCHIVE_BYTES:
        OUTPUT_PATH.unlink()
        raise ValueError("Submission exceeds the maximum file size")

    with ZipFile(OUTPUT_PATH) as archive:
        notebook_files = [
            name for name in archive.namelist()
            if name.endswith(".ipynb")
        ]

        if len(notebook_files) != 1:
            raise ValueError(
                "Submission must contain exactly one notebook"
            )

        if archive.testzip() is not None:
            raise ValueError("Submission ZIP failed integrity check")

    print(f"Submission created: {OUTPUT_PATH}")
    print(f"Archive size: {archive_size / 1_000_000:.2f} MB")
    print(f"Python source files included: {len(source_files)}")
    print("Notebook execution and ZIP integrity checks passed.")


if __name__ == "__main__":
    build_submission()