"""Execute a notebook in place with jupyter_client (usage: python scripts/run_notebook.py notebooks/X.ipynb)."""
import json
import sys
from pathlib import Path
from queue import Empty

from jupyter_client.manager import start_new_kernel

path = Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
count = 0
failed = False
try:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        msg_id = kc.execute("".join(cell["source"]))
        outputs = []
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=900)
            except Empty:
                raise RuntimeError("timeout")
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "stream":
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == content["name"]:
                    outputs[-1]["text"] += content["text"]
                else:
                    outputs.append({"output_type": "stream", "name": content["name"], "text": content["text"]})
            elif kind in ("display_data", "execute_result"):
                out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                if kind == "execute_result":
                    out["execution_count"] = count
                outputs.append(out)
            elif kind == "error":
                outputs.append({"output_type": "error", "ename": content["ename"], "evalue": content["evalue"],
                                "traceback": content["traceback"]})
                failed = True
        for out in outputs:
            if out["output_type"] == "stream":
                out["text"] = out["text"].splitlines(keepends=True)
            elif "data" in out:
                out["data"] = {k: (v.splitlines(keepends=True) if isinstance(v, str) and not k.startswith("image/") else v)
                               for k, v in out["data"].items()}
        cell["outputs"] = outputs
        cell["execution_count"] = count
        if failed:
            print("ERROR in cell", count)
            print("\n".join(o.get("evalue", "") for o in outputs if o["output_type"] == "error"))
            break
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("executed", count, "cells", "FAILED" if failed else "ok")
