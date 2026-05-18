import os
import subprocess
import tempfile

def execute_python(code: str, plot_path: str) -> dict:
    """
    Execute `code` in a fresh subprocess.
    Injects:  PLOT_PATH = r'<plot_path>'
    as the very first line so the LLM can always use PLOT_PATH directly.

    Returns:
        {"stdout": str, "stderr": str, "returncode": int}
    """
    injected = f"PLOT_PATH = r'{plot_path}'\n" + code

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(injected)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "stdout":     result.stdout.strip(),
            "stderr":     result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":     "",
            "stderr":     "Execution timed out after 30 seconds.",
            "returncode": 1,
        }
    except Exception as e:
        return {
            "stdout":     "",
            "stderr":     f"Execution error: {e}",
            "returncode": 1,
        }
    finally:
        os.unlink(tmp_path)
