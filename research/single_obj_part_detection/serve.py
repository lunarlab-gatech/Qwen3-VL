import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part_detection"))
from prompt import Model

_CHECKPOINT = Path.home() / "Qwen3-VL" / "models" / Model.QWEN3_VL_235B_A22B.value

# Replace this process with the vLLM OpenAI-compatible server.
# To increase throughput later: raise --max-num-seqs and batch requests in LLM.py.
os.execvp(sys.executable, [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model",                str(_CHECKPOINT),
    "--served-model-name",    "qwen3-vl",
    "--tensor-parallel-size", "4",
    "--pipeline-parallel-size", "2",
    "--gpu-memory-utilization", "0.95",
    "--max-model-len",        "7168",
    "--max-num-seqs",         "5",
    "--trust-remote-code",
    "--host",                 "0.0.0.0",
    "--port",                 "8000",
])
