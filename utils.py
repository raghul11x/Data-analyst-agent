import base64
import json
import os
import re

import pandas as pd

def extract_code_blocks(text: str) -> list:
    """Return every ```python … ``` or ``` … ``` block found in `text`, in order."""
    blocks, pos = [], 0
    while pos < len(text):
        if "```python" in text[pos:]:
            s = text.find("```python", pos) + 9
            e = text.find("```", s)
            if e == -1:
                break
            blocks.append(text[s:e].strip())
            pos = e + 3
        elif "```" in text[pos:]:
            s = text.find("```", pos) + 3
            e = text.find("```", s)
            if e == -1:
                break
            blocks.append(text[s:e].strip())
            pos = e + 3
        else:
            break
    return blocks

def strip_code_blocks(text: str) -> str:
    """Remove all ``` … ``` fences from a string (used to clean the final report)."""
    return re.sub(r'```[\s\S]*?```', '', text).strip()

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def grab_plot(plot_path: str):
    """
    If `plot_path` exists, read it, base64-encode it, delete it, and return the string.
    Returns None if the file doesn't exist.
    """
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(plot_path)
        return b64
    return None

def load_dataset(file_storage):
    """
    Accept a Flask FileStorage object.
    Returns (DataFrame, info_string).
    Raises ValueError for unsupported file types.
    """
    import tempfile as _tmp, os as _os
    fname = file_storage.filename.lower()
    if fname.endswith(".csv"):
        df = pd.read_csv(file_storage, low_memory=False)
    elif fname.endswith((".xlsx", ".xls")):
        suffix = ".xlsx" if fname.endswith(".xlsx") else ".xls"
        raw = _tmp.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            file_storage.save(raw.name)
            raw.close()
            if suffix == ".xlsx":
                for engine in ["calamine", "openpyxl"]:
                    try:
                        df = pd.read_excel(raw.name, engine=engine)
                        break
                    except Exception:
                        continue
                else:
                    df = pd.read_excel(raw.name)
            else:
                df = pd.read_excel(raw.name, engine="xlrd")
        finally:
            try:
                _os.unlink(raw.name)
            except Exception:
                pass
    else:
        raise ValueError("Unsupported format — upload CSV or Excel (.xlsx / .xls).")

    info = (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns: {list(df.columns)}\n"
        f"Data types:\n{df.dtypes.to_string()}\n"
        f"Null counts: {df.isnull().sum().to_dict()}\n"
        f"Numeric summary:\n{df.describe().to_string()}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_string()}"
    )
    return df, info

def load_multiple_datasets(files):
    """
    Accept a list of Flask FileStorage objects.
    Returns (dict of name->path, combined info string).
    Each file is loaded, saved to a temp CSV, and described.
    """
    import tempfile, os
    datasets = {}
    info_parts = []

    for file_storage in files:
        fname = file_storage.filename.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file_storage)
        elif fname.endswith(".xlsx"):
            df = pd.read_excel(file_storage, engine="openpyxl")
        elif fname.endswith(".xls"):
            df = pd.read_excel(file_storage, engine="xlrd")
        else:
            raise ValueError(f"Unsupported format for {file_storage.filename}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)
        tmp.close()

        datasets[file_storage.filename] = tmp.name

        info_parts.append(
            f"--- Dataset: {file_storage.filename} ---\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {list(df.columns)}\n"
            f"Data types:\n{df.dtypes.to_string()}\n"
            f"Null counts: {df.isnull().sum().to_dict()}\n"
            f"Numeric summary:\n{df.describe().to_string()}\n"
            f"Sample (first 3 rows):\n{df.head(3).to_string()}"
        )

    combined_info = "\n\n".join(info_parts)
    return datasets, combined_info
