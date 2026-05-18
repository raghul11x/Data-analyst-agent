import json
import time

import requests as req_lib

from config import API_KEY, API_URL, MODEL, SESSIONS
from executor import execute_python
from utils import extract_code_blocks, strip_code_blocks, sse, grab_plot

import re as _re

_THINK_PATTERNS = _re.compile(
    r"(the (instructions?|rules?|prompt) (say|says|state|tell)|"
    r"do not narrate|do NOT|must not|we must|we need to|we are (given|told|asked)|"
    r"let me|let's|i (will|am going to|need to|should|must|can)|"
    r"note that|important:|actually,|however,|but note|so we|"
    r"since (the|we|this)|first,|then,|next,|finally,|"
    r"possible questions|we are to output)",
    _re.IGNORECASE
)

def _sanitise(text: str) -> str:
    """
    Remove lines that are the model thinking out loud / narrating.
    Keeps code blocks and lines that look like genuine analysis prose.
    """
    lines = text.split("\n")
    clean = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            clean.append(line)
            continue
        if in_code:
            clean.append(line)
            continue

        if line.startswith(("#", "-", "*", "  -", "  *")):
            clean.append(line)
            continue

        if _THINK_PATTERNS.search(line):
            continue
        clean.append(line)
    return "\n".join(clean).strip()

SYSTEM_PROMPT = """You are a senior data analyst. Analyse the CSV dataset provided.

IMPORTANT — Handle user typos and geographic context intelligently:
- Always use case-insensitive matching: df[col].str.contains(term, case=False, na=False)
- Interpret geographic intent: if user says "Karnataka leads", search for Bangalore, Bengaluru, Mysore, Hubli etc.
- If user says "Tamil Nadu leads", search for Chennai, Coimbatore, Madurai etc.
- If user says "Maharashtra leads", search for Mumbai, Pune, Nagpur etc.
- Fix common typos automatically: "karntaka"→Karnataka, "bangalor"→Bangalore, "chenai"→Chennai
- If exact filter returns 0 results, broaden the search or try related terms
- Always explain what you searched for in your findings

FORMAT FOR EVERY RESPONSE (follow exactly):
One short sentence. Then a ```python code block. Nothing else.

For plots: plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150) then plt.close()

When you have enough real results from code output, write the final report.
The final report MUST use this exact structure with real numbers:

## Final Report

### Key Findings
- **Insight label**: One sentence with actual numbers from the data.
- **Insight label**: One sentence with actual numbers from the data.
- **Insight label**: One sentence with actual numbers from the data.

### Trends
- **Pattern label**: One sentence describing the trend with actual numbers.
- **Pattern label**: One sentence describing the trend with actual numbers.

### Recommendations
- **Action label**: One sentence recommendation based on findings.
- **Action label**: One sentence recommendation based on findings.

RULES:
- Every bullet starts with - (hyphen space)
- Every bullet has a **bold label** followed by a colon
- Use only numbers you actually saw in code output
- No code blocks inside the final report
- No meta-commentary, no explaining the rules"""

SUGGEST_SYSTEM = """Suggest 5 specific data analysis questions for the given dataset columns.
Output ONLY a valid JSON array of 5 strings. Nothing else.
Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]"""

FOLLOWUP_SYSTEM = """You are a data analyst. Answer the question directly using actual data values.
Handle typos and spelling mistakes — use case-insensitive fuzzy matching:
  df[col].str.contains(term, case=False, na=False)
If no results found with exact term, try partial matches or common variations.
If you need code, write one ```python block. PLOT_PATH is pre-defined.
Save plots: plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150) then plt.close()
Be concise. One paragraph maximum. No meta-commentary."""

def _post(messages: list, max_tokens: int, timeout: int):
    return req_lib.post(
        url=API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5050",
            "X-Title": "Agentic Analyser",
        },
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
        timeout=timeout,
    )

def call_llm(messages: list, system: str = None, max_tokens: int = 2000) -> str:
    """Standard LLM call with retry on rate-limit."""
    sys_msg = system or SYSTEM_PROMPT
    full = [{"role": "system", "content": sys_msg}] + messages
    for attempt in range(3):
        try:
            resp = _post(full, max_tokens, timeout=45)
            print(f"[llm] HTTP {resp.status_code}")
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[llm] rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                return f"Error {resp.status_code}: {resp.text[:100]}"
            try:
                data = resp.json()
            except Exception:
                time.sleep(5)
                continue
            if "error" in data:
                msg  = data["error"].get("message", "")
                code = str(data["error"].get("code", ""))
                if "rate" in msg.lower() or "429" in code or "quota" in msg.lower():
                    wait = 30 * (attempt + 1)
                    time.sleep(wait)
                    continue
                return f"API Error: {msg}"
            return data["choices"][0]["message"]["content"]
        except req_lib.exceptions.Timeout:
            print(f"[llm] timeout attempt {attempt + 1}")
            time.sleep(5)
        except Exception as e:
            print(f"[llm] exception: {e}")
            time.sleep(5)
    return "Max retries reached."

def call_llm_fast(messages: list, system: str = None) -> str:
    """Low-token call — used for question suggestions."""
    sys_msg = system or SUGGEST_SYSTEM
    full = [{"role": "system", "content": sys_msg}] + messages
    for attempt in range(2):
        try:
            resp = _post(full, max_tokens=300, timeout=20)
            if resp.status_code == 429:
                time.sleep(20)
                continue
            if resp.status_code >= 400:
                return ""
            data = resp.json()
            if "error" in data:
                return ""
            return data["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(3)
    return ""

def run_codes(code_blocks: list, plot_path: str):
    """
    Execute every code block; merge outputs.
    Returns: (tool_txt, status_str, plot_b64_or_None)
    """
    stdout_parts, stderr_parts, last_status = [], [], "Success"
    for code in code_blocks:
        r = execute_python(code, plot_path)
        if r["stdout"]:  stdout_parts.append(r["stdout"])
        if r["stderr"]:  stderr_parts.append(r["stderr"])
        if r["returncode"] != 0:
            last_status = "Error"

    stdout   = "\n".join(stdout_parts) or "(no output)"
    stderr   = "\n".join(stderr_parts)
    tool_txt = f"Code execution result:\nStatus: {last_status}\nOutput:\n{stdout}"
    if stderr:
        tool_txt += f"\nErrors:\n{stderr}"

    plot_b64 = grab_plot(plot_path)
    return tool_txt, last_status, plot_b64

def run_agent_stream(session_id, dataset_paths, dataset_info, question, max_iter, q):
    """
    dataset_paths: dict of {filename: tmp_csv_path} or a single path string (legacy).
    Uses SQLite DB for all message/plot/report persistence.
    """
    import os
    try:
        _run_agent_stream_inner(session_id, dataset_paths, dataset_info, question, max_iter, q)
    except Exception as e:
        print(f"[agent] FATAL ERROR in run_agent_stream: {e}")
        import traceback; traceback.print_exc()
        q.put(sse("error", {"message": str(e)}))
        q.put(sse("done", {}))
        q.put(None)

def _run_agent_stream_inner(session_id, dataset_paths, dataset_info, question, max_iter, q):
    import os

    if isinstance(dataset_paths, str):
        dataset_paths = {"dataset.csv": dataset_paths}

    session   = SESSIONS[session_id]
    plot_path = session["plot_path"]

    if os.path.exists(plot_path):
        os.remove(plot_path)

    paths_block = "\n".join(
        f"  {name}: r\"{path}\"" for name, path in dataset_paths.items()
    )
    if len(dataset_paths) == 1:
        primary = list(dataset_paths.values())[0]
        paths_intro = f"Dataset path: r\"{primary}\""
    else:
        paths_intro = (
            f"You have {len(dataset_paths)} datasets. Load each with pd.read_csv(path):\n"
            + paths_block
            + "\nYou can merge, compare, or analyse them independently."
        )

    first_msg = (
        f"{paths_intro}\n"
        f"PLOT_PATH is pre-injected — just call plt.savefig(PLOT_PATH, ...)\n"
        f"{dataset_info}\n\n"
        f"User question: {question}\n\n"
        "Respond in plain English + ```python code blocks only. No JSON. Begin."
    )
    session["messages"].append({"role": "user", "content": first_msg})

    for iteration in range(max_iter):
        q.put(sse("thinking", {
            "iteration": iteration + 1,
            "total": max_iter,
            "label": f"Iteration {iteration + 1} of {max_iter}…",
        }))
        resp = _sanitise(call_llm([m for m in session["messages"] if m["role"] != "system"]))
        session["messages"].append({"role": "assistant", "content": resp})

        if "## Final Report" in resp or "## Key Findings" in resp:
            plot_b64 = grab_plot(plot_path)
            if plot_b64:
                session["plots"].append(plot_b64)
            clean = strip_code_blocks(resp)
            session["final_report"] = clean
            q.put(sse("final_report", {"content": clean, "plot": plot_b64}))
            q.put(sse("done", {}))
            q.put(None)
            return

        stripped = resp.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            session["messages"].append({"role": "user", "content": "Stop using JSON. Reply in plain English + ```python blocks only."})
            q.put(sse("agent", {"content": resp, "iteration": iteration + 1}))
            continue

        q.put(sse("agent", {"content": resp, "iteration": iteration + 1}))

        blocks = extract_code_blocks(resp)
        if blocks:
            tool_txt, status, plot_b64 = run_codes(blocks, plot_path)
            if plot_b64:
                session["plots"].append(plot_b64)
            q.put(sse("tool", {"content": tool_txt, "status": status, "plot": plot_b64}))
            session["messages"].append({"role": "user", "content": tool_txt})
        else:
            session["messages"].append({"role": "user", "content": "Continue analysis or write ## Final Report."})

    q.put(sse("thinking", {
        "iteration": max_iter,
        "total": max_iter,
        "label": "Compiling final report…",
    }))
    session["messages"].append({"role": "user", "content": "Write ## Final Report now — plain markdown only, no code blocks."})
    final    = call_llm([m for m in session["messages"] if m["role"] != "system"])
    plot_b64 = grab_plot(plot_path)
    if plot_b64:
        session["plots"].append(plot_b64)
    clean = strip_code_blocks(final)
    session["final_report"] = clean
    q.put(sse("final_report", {"content": clean, "plot": plot_b64}))
    q.put(sse("done", {}))
    q.put(None)

def run_followup_stream(session_id, question, q):
    try:
        _run_followup_inner(session_id, question, q)
    except Exception as e:
        print(f"[agent] FATAL ERROR in run_followup_stream: {e}")
        q.put(sse("error", {"message": str(e)}))
        q.put(sse("done", {}))
        q.put(None)

def _run_followup_inner(session_id, question, q):
    session   = SESSIONS[session_id]
    plot_path = session["plot_path"]
    session["messages"].append({"role": "user", "content": question})

    q.put(sse("thinking", {"iteration": 1, "total": 3, "label": "Thinking…"}))
    resp = call_llm(
        [m for m in session["messages"] if m["role"] != "system"],
        system=FOLLOWUP_SYSTEM,
    )
    session["messages"].append({"role": "assistant", "content": resp})

    blocks = extract_code_blocks(resp)
    if blocks:
        q.put(sse("thinking", {"iteration": 2, "total": 3, "label": "Running code…"}))
        tool_txt, status, plot_b64 = run_codes(blocks, plot_path)
        if plot_b64:
            session["plots"].append(plot_b64)
        q.put(sse("tool", {"content": tool_txt, "status": status, "plot": plot_b64}))
        session["messages"].append({"role": "user", "content": tool_txt})

        q.put(sse("thinking", {"iteration": 3, "total": 3, "label": "Formulating answer…"}))
        final = call_llm(
            [m for m in session["messages"] if m["role"] != "system"],
            system=FOLLOWUP_SYSTEM,
        )
        session["messages"].append({"role": "assistant", "content": final})
        q.put(sse("followup_answer", {"content": final}))
    else:
        q.put(sse("followup_answer", {"content": resp}))

    q.put(sse("done", {}))
    q.put(None)
