import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PICKLE_STUB_PREFIXES = ("roman", "robotdatapy")

class _PickleStubFinder:
    """Catch-all import hook for modules only needed for pickle deserialization."""
    def find_spec(self, name, path, target=None):
        if not any(name == p or name.startswith(p + ".") for p in _PICKLE_STUB_PREFIXES):
            return None
        import importlib.util
        spec = importlib.util.spec_from_loader(name, self)
        spec.submodule_search_locations = []
        return spec

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__file__ = None
        mod.__spec__ = spec
        mod.__getattr__ = lambda attr: type(attr, (), {})
        return mod

    def exec_module(self, module):
        pass

sys.meta_path.append(_PickleStubFinder())

import json
import os
import re
import tempfile
import time
import pickle
import cv2
import numpy as np
import torch
from enum import Enum
from pathlib import Path
from PIL import Image as PilImage
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn


class Model(Enum):
    QWEN25_VL_3B     = "Qwen2.5-VL-3B-Instruct"
    QWEN25_VL_3B_AWQ = "Qwen2.5-VL-3B-Instruct-AWQ"
    QWEN3_VL_8B_FP8  = "Qwen3-VL-8B-Instruct-FP8"
    QWEN3_VL_30B_FP8 = "Qwen3-VL-30B-A3B-Instruct-FP8"


_BOX_COLORS = [
    (0, 255, 0), (255, 80, 0), (0, 80, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
]

_SCRIPT_DIR = Path(__file__).resolve().parent


# ── Data loading ─────────────────────────────────────────────────────────────

def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text().strip()


def load_roman_map(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_segments(roman_map) -> list:
    segs = roman_map.segments
    if isinstance(segs, dict):
        return list(segs.values())
    return list(segs)


# ── Semantic label lookup ─────────────────────────────────────────────────────

def load_wordnet_features(files_dir: Path) -> tuple[np.ndarray, list[str]] | None:
    feat_path = files_dir / "synset_features.npy"
    word_path = files_dir / "synset_list.npy"
    if not feat_path.exists() or not word_path.exists():
        return None
    features = np.load(str(feat_path)).astype(np.float32)
    synset_names: list[str] = np.load(str(word_path), allow_pickle=True).tolist()
    return features, synset_names


def descriptor_to_synset_name(
    descriptor: np.ndarray,
    features: np.ndarray,
    synset_names: list[str],
) -> str:
    emb = descriptor.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    similarities = features @ emb
    return synset_names[int(np.argmax(similarities))]


def synset_to_display_word(synset_name: str) -> str:
    """Return the primary lemma name (e.g. 'car' from 'car.n.01')."""
    return wn.synset(synset_name).lemma_names()[0].replace('_', ' ')


# ── Meronym lookup ────────────────────────────────────────────────────────────

def _wn_neighbors(syn, direction: str, depth: int) -> list:
    """BFS over hyponyms or hypernyms up to `depth` levels."""
    seen, frontier = set(), [syn]
    for _ in range(depth):
        nxt = []
        for s in frontier:
            for h in getattr(s, direction)():
                if h not in seen:
                    seen.add(h)
                    nxt.append(h)
        frontier = nxt
    return list(seen)


def get_meronyms(synset_name: str) -> list[str]:
    """Return sorted part-meronym synset names for synset_name and its hypo/hypernyms."""
    syn = wn.synset(synset_name)
    search = [syn] + _wn_neighbors(syn, 'hyponyms', 4) + _wn_neighbors(syn, 'hypernyms', 4)
    mset: set = set()
    for s in search:
        mset.update(s.part_meronyms())
    return sorted(s.name() for s in mset)


def build_reference_list(meronyms: list[str]) -> str:
    if not meronyms:
        return "(no parts found)"
    return ", ".join(f"[{i + 1}] {name}" for i, name in enumerate(meronyms))


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(template: str, word: str, reference_list: str) -> str:
    return template.replace("<word>", word).replace("<reference_list>", reference_list)


# ── Keyframe extraction ───────────────────────────────────────────────────────

def get_keyframe_image(segment) -> np.ndarray | None:
    """Return image_crop (RGB uint8 ndarray) from the first valid keyframe bin."""
    for kb in (segment.keyframe_bins or []):
        kf = kb.best_full if kb.best_full is not None else kb.best_clipped
        if kf is not None and kf.image_crop is not None:
            return kf.image_crop
    return None


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> tuple[LLM, AutoProcessor]:
    model = LLM(
        model=str(checkpoint_path),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=7168,
        max_num_seqs=1,
        seed=0,
    )
    processor = AutoProcessor.from_pretrained(str(checkpoint_path))
    return model, processor


def build_inputs(image_rgb: np.ndarray, prompt: str, processor: AutoProcessor) -> dict:
    """Build vLLM input dict from an RGB numpy array and a filled prompt string."""
    pil_img = PilImage.fromarray(image_rgb)

    # Bounds check against processor's max_pixels
    ip = processor.image_processor
    max_pixels = getattr(ip, 'max_pixels', None) or ip.size.get("longest_edge", float('inf'))
    orig_w, orig_h = pil_img.size
    assert orig_w * orig_h <= max_pixels, (
        f"Image {orig_w}x{orig_h} ({orig_w * orig_h} px) exceeds max_pixels={max_pixels}"
    )

    # Save to temp file (process_vision_info expects a path or URL)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    pil_img.save(tmp_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    os.unlink(tmp_path)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


# ── Response parsing / validation ─────────────────────────────────────────────

_NOT_A_RE = re.compile(r"^Not a .+, but a (.+)\.$", re.IGNORECASE)

def parse_response(response: str) -> tuple[list | None, str]:
    """Parse model output as a JSON array of {b, l} detection dicts.
    Also handles the pre-check format: 'Not a <word>, but a <actual>.'"""
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return None, "Expected a JSON array at the top level"
        return data, ""
    except json.JSONDecodeError:
        pass
    m = _NOT_A_RE.match(text)
    if m:
        return [], ""
    return None, "LVLM output not in valid format"


def validate_response(data: list) -> tuple[list, list[str]]:
    """Validate each detection entry; return (valid_list, warnings)."""
    warnings: list[str] = []
    result: list = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "b" not in item or "l" not in item:
            warnings.append(f"[{i}]: missing 'b' or 'l' field")
            continue
        b = item["b"]
        if not (isinstance(b, list) and len(b) == 4
                and all(isinstance(v, (int, float)) for v in b)):
            warnings.append(f"[{i}] '{item.get('l', '?')}': invalid 'b' (need [ymin,xmin,ymax,xmax])")
            continue
        result.append(item)
    return result, warnings


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw boxes on a BGR image. Box format: [ymin, xmin, ymax, xmax] in [0, 1000]."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for i, item in enumerate(detections):
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        ymin, xmin, ymax, xmax = item["b"]
        x1 = int(xmin / 1000 * w)
        y1 = int(ymin / 1000 * h)
        x2 = int(xmax / 1000 * w)
        y2 = int(ymax / 1000 * h)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = item["l"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Draw label background above box (flip below if clipped)
        if y1 - th - 6 >= 0:
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        else:
            cv2.rectangle(out, (x1, y2), (x1 + tw + 4, y2 + th + 6), color, -1)
            cv2.putText(out, label, (x1 + 2, y2 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def _render_text_block(canvas, text: str, color, x: int, y: int,
                       font, line_h: int, chars_per_line: int = 72) -> int:
    """Word-wrap text onto an unbounded canvas, returning the new y position."""
    current = ""
    for tok in text.split():
        if len(current) + len(tok) + 1 > chars_per_line:
            cv2.putText(canvas, current, (x, y), font, 0.45, color, 1)
            y += line_h
            current = tok
        else:
            current = (current + " " + tok).strip()
    if current:
        cv2.putText(canvas, current, (x, y), font, 0.45, color, 1)
        y += line_h
    return y


def display_result(
    image_rgb: np.ndarray,
    seg_id: int,
    word: str,
    prompt: str,
    response: str,
    elapsed_sec: float,
    detections: list,
    parse_error: str,
) -> bool:
    """Show annotated image + scrollable response panel. Returns False if user presses 'q'."""
    TARGET_H = 600

    # Scale up tiny crops with nearest-neighbour so pixels stay crisp
    h, w = image_rgb.shape[:2]
    scale = max(1, TARGET_H // max(h, 1))
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Final rescale to exactly TARGET_H height
    h, w = img.shape[:2]
    if h != TARGET_H:
        ratio = TARGET_H / h
        img = cv2.resize(img, (int(w * ratio), TARGET_H))
    h, w = img.shape[:2]

    img = draw_detections(img, detections)

    # Render all text onto a tall virtual canvas, then scroll a window over it
    panel_w = 700
    font, line_h, x = cv2.FONT_HERSHEY_SIMPLEX, 18, 10
    virtual_h = 4000
    canvas = np.full((virtual_h, panel_w, 3), 30, dtype=np.uint8)
    y = 24

    status = f"[{len(detections)} detections]" if not parse_error else "[INVALID JSON]"
    header_color = (100, 200, 255) if not parse_error else (60, 60, 255)
    for line in [f"Segment {seg_id}: {word}", f"Inference: {elapsed_sec:.2f}s  {status}", ""]:
        cv2.putText(canvas, line, (x, y), font, 0.45, header_color, 1)
        y += line_h

    cv2.putText(canvas, "-- PROMPT --", (x, y), font, 0.45, (160, 160, 80), 1)
    y += line_h
    y = _render_text_block(canvas, prompt, (180, 180, 100), x, y, font, line_h)
    y += line_h // 2

    cv2.putText(canvas, "-- RESPONSE --", (x, y), font, 0.45, (160, 160, 80), 1)
    y += line_h
    body = parse_error if parse_error else response
    y = _render_text_block(canvas, body, (220, 220, 220), x, y, font, line_h)

    content_h = y + line_h
    scroll_max = max(0, content_h - h)
    scroll = 0
    scroll_step = line_h * 3
    win = "Part Detection  [arrows = scroll | any key = next | q = quit]"

    while True:
        panel = canvas[scroll:scroll + h].copy()
        # Scroll indicator
        if scroll_max > 0:
            bar_h = max(20, int(h * h / content_h))
            bar_y = int(scroll / scroll_max * (h - bar_h))
            cv2.rectangle(panel, (panel_w - 6, bar_y), (panel_w - 2, bar_y + bar_h), (120, 120, 120), -1)
        cv2.imshow(win, np.hstack([img, panel]))
        key = cv2.waitKeyEx(0)
        if key == ord("q"):
            return False
        elif key in (65364, 2621440):  # down arrow (Linux / Windows)
            scroll = min(scroll + scroll_step, scroll_max)
        elif key in (65362, 2490368):  # up arrow
            scroll = max(scroll - scroll_step, 0)
        else:
            return True


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(
    model: LLM,
    processor: AutoProcessor,
    segments: list,
    prompt_template: str,
    wn_features: np.ndarray | None,
    wn_synset_names: list[str] | None,
) -> None:
    sampling_params = SamplingParams(max_tokens=1024)

    for segment in segments:
        image_rgb = get_keyframe_image(segment)
        if image_rgb is None:
            print(f"  [segment {segment.id}] no keyframe image, skipping")
            continue

        # Resolve semantic label
        desc = getattr(segment, 'semantic_descriptor', None)
        if desc is not None and wn_features is not None:
            synset_name = descriptor_to_synset_name(desc, wn_features, wn_synset_names)
            word = synset_to_display_word(synset_name)
        else:
            synset_name = None
            word = f"segment_{segment.id}"

        # Build dynamic prompt
        meronyms = get_meronyms(synset_name) if synset_name else []
        ref_list = build_reference_list(meronyms)
        prompt = build_prompt(prompt_template, word, ref_list)

        inputs = build_inputs(image_rgb, prompt, processor)

        t0 = time.perf_counter()
        outputs = model.generate(inputs, sampling_params=sampling_params)
        elapsed = time.perf_counter() - t0

        response = outputs[0].outputs[0].text
        print(f"[segment {segment.id} / {word}] ({elapsed:.2f}s) {response}")

        parse_error = ""
        parsed, parse_error = parse_response(response)
        detections: list = []
        if parsed is not None:
            detections, warnings = validate_response(parsed)
            for w_msg in warnings:
                print(f"  [validation] {w_msg}")
        else:
            print(f"  [parse error] {parse_error}")

        if not display_result(image_rgb, segment.id, word, prompt, response, elapsed, detections, parse_error):
            break

    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    home = Path.home()
    model_choice  = Model.QWEN3_VL_8B_FP8
    checkpoint_path = home / "Qwen3-VL" / "models" / model_choice.value
    pkl_path      = _SCRIPT_DIR / "Husky1.pkl"
    prompt_path   = _SCRIPT_DIR / "PROMPT.md"
    files_dir     = _SCRIPT_DIR / "files"

    prompt_template = load_prompt(prompt_path)

    roman_map = load_roman_map(pkl_path)
    segments  = get_segments(roman_map)
    print(f"Loaded {len(segments)} segments from {pkl_path.name}")

    wn_result = load_wordnet_features(files_dir)
    if wn_result is None:
        raise FileNotFoundError(
            f"Pre-computed WordNet features not found in {files_dir}."
        )
    wn_features, wn_synset_names = wn_result
    print(f"Loaded {len(wn_synset_names)} synsets from {files_dir}")

    model, processor = load_model(checkpoint_path)
    run_inference(model, processor, segments, prompt_template, wn_features, wn_synset_names)


if __name__ == "__main__":
    main()
