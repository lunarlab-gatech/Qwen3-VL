import json
import re
import time
import cv2
import numpy as np
import torch
from decimal import Decimal
from enum import Enum
from pathlib import Path
from PIL import Image as PilImage
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from robotdataprocess import ImageDataOnDisk

class Model(Enum):
    QWEN25_VL_3B       = "Qwen2.5-VL-3B-Instruct"
    QWEN25_VL_3B_AWQ   = "Qwen2.5-VL-3B-Instruct-AWQ"
    QWEN3_VL_8B_FP8    = "Qwen3-VL-8B-Instruct-FP8"
    QWEN3_VL_30B_FP8   = "Qwen3-VL-30B-A3B-Instruct-FP8"


_BOX_COLORS = [
    (0, 255, 0), (255, 80, 0), (0, 80, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
]


def load_prompt(prompt_path: Path) -> str:
    """Load the prompt template from a markdown file.

    Args:
        prompt_path: Path to the markdown file containing the prompt template.

    Returns:
        Prompt template string with leading/trailing whitespace stripped.
    """
    return prompt_path.read_text().strip()


def load_image_paths(dataset_path: Path, robot_name: str, stride: int) -> list[Path]:
    """Load and crop RGB image paths for the given robot from the V2.4.C dataset.

    Args:
        dataset_path: Root data directory of the dataset.
        robot_name: Name of the robot (e.g. 'Husky1').
        stride: Step size for subsampling frames.

    Returns:
        List of image file paths in chronological order.
    """

    rgb_data = ImageDataOnDisk.from_image_files(
        dataset_path / robot_name / "rgb_stereo_left", "front_center_Scene"
    )
    rgb_data.crop_data(Decimal("0.0"), Decimal("382.85"))
    return rgb_data.images.image_paths[::stride]


def load_model(checkpoint_path: Path) -> tuple[LLM, AutoProcessor]:
    """Load the vLLM model and processor from the given checkpoint.

    The processor converts raw inputs (text and images) into tensors the model
    expects. The model runs GPU inference over those tensors to generate tokens.

    Args:
        checkpoint_path: Path to the model checkpoint directory.

    Returns:
        Tuple of (model, processor).
    """
    model = LLM(
        model=str(checkpoint_path),
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=4096,
        max_num_seqs=1,
        seed=0,
    )
    processor = AutoProcessor.from_pretrained(str(checkpoint_path))
    return model, processor


def format_tracked_objects(tracked: dict[int, dict]) -> str:
    """Format the current tracking state as a human-readable string for the prompt.

    Args:
        tracked: Mapping of tracking ID to object dict with 'label' and 'bounding_box'.

    Returns:
        Formatted string listing each tracked object, or '(none)' if empty.
    """
    if not tracked:
        return "(none)"
    return "\n".join(
        f"  ID {tid}: {obj['label']} at {obj['bounding_box']}"
        for tid, obj in tracked.items()
    )


def build_inputs(
    image_path: Path,
    prompt_template: str,
    processor: AutoProcessor,
    tracked: dict[int, dict],
) -> dict:
    """Build the vLLM input dict for a single image, injecting the current tracking state.

    Args:
        image_path: Path to the image file.
        prompt_template: Prompt string with a {tracked_objects} placeholder.
        processor: The model's AutoProcessor for chat template application.
        tracked: Current tracking state used to fill the prompt placeholder.

    Returns:
        Dict with 'prompt', 'multi_modal_data', and 'mm_processor_kwargs' keys.
    """
    with PilImage.open(image_path) as pil_img:
        orig_w, orig_h = pil_img.size
    ip = processor.image_processor
    max_pixels = ip.max_pixels if ip.max_pixels is not None else ip.size["longest_edge"]
    assert orig_w * orig_h <= max_pixels, (
        f"{image_path.name}: original resolution {orig_w}x{orig_h} "
        f"({orig_w * orig_h} px) exceeds max_pixels={max_pixels}; "
        f"model sees a rescaled image and bounding box coordinates will be misaligned"
    )

    prompt = prompt_template.replace("{tracked_objects}", format_tracked_objects(tracked))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
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

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def parse_response(response: str) -> tuple[dict | None, str]:
    """Parse VLM response as a JSON object with 'new', 'tracked', and 'lost' keys.

    Handles markdown code fences and Python tuple syntax (x, y) as a fallback.

    Args:
        response: Raw text output from the model.

    Returns:
        Tuple of (parsed dict or None, error message string).
    """
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

    for attempt in (text, re.sub(r"\((\s*-?\d+\s*,\s*-?\d+\s*)\)", r"[\1]", text)):
        try:
            data = json.loads(attempt)
            if not isinstance(data, dict):
                return None, "Expected a JSON object at the top level"
            return data, ""
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse response as JSON: {response[:120]!r}"


def _check_bbox_bounds(bb: list, img_w: int, img_h: int, label: str) -> None:
    x1, y1, x2, y2 = bb
    if not (0 <= x1 <= img_w and 0 <= x2 <= img_w and 0 <= y1 <= img_h and 0 <= y2 <= img_h):
        raise ValueError(
            f"Bounding box {bb} for '{label}' is outside image bounds ({img_w}x{img_h})"
        )


def validate_response(data: dict, tracked: dict[int, dict], img_w: int, img_h: int) -> tuple[dict, list[str]]:
    """Validate the parsed tracking response against the expected schema.

    Entries in 'tracked' or 'lost' that reference IDs not present in the current
    tracking state are discarded, since the model may hallucinate IDs.

    Args:
        data: Raw parsed dict from parse_response.
        tracked: Current tracking state used to cross-check reported IDs.
        img_w: Image width in pixels, used to bounds-check bounding boxes.
        img_h: Image height in pixels, used to bounds-check bounding boxes.

    Returns:
        Tuple of (validated dict with 'new', 'tracked', 'lost' lists,
                  list of warning strings for skipped entries).
    """
    warnings = []
    result: dict = {"new": [], "tracked": [], "lost": []}

    for i, obj in enumerate(data.get("new", [])):
        if not isinstance(obj, dict) or "label" not in obj or "bounding_box" not in obj:
            warnings.append(f"new[{i}]: missing 'label' or 'bounding_box'")
            continue
        bb = obj["bounding_box"]
        if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb)):
            warnings.append(f"new[{i}] '{obj['label']}': invalid bounding_box (need [x1,y1,x2,y2])")
            continue
        _check_bbox_bounds(bb, img_w, img_h, obj["label"])
        result["new"].append(obj)

    for i, obj in enumerate(data.get("tracked", [])):
        if not isinstance(obj, dict) or "id" not in obj or "bounding_box" not in obj:
            warnings.append(f"tracked[{i}]: missing 'id' or 'bounding_box'")
            continue
        tid = obj["id"]
        if tid not in tracked:
            warnings.append(f"tracked[{i}]: id={tid} not in current tracking state, ignoring")
            continue
        bb = obj["bounding_box"]
        if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb)):
            warnings.append(f"tracked[{i}] id={tid}: invalid bounding_box (need [x1,y1,x2,y2])")
            continue
        _check_bbox_bounds(bb, img_w, img_h, f"id={tid}")
        result["tracked"].append(obj)

    for i, tid in enumerate(data.get("lost", [])):
        if not isinstance(tid, int):
            warnings.append(f"lost[{i}]: expected integer ID, got {tid!r}")
            continue
        if tid not in tracked:
            warnings.append(f"lost[{i}]: id={tid} not in current tracking state, ignoring")
            continue
        result["lost"].append(tid)

    return result, warnings


def update_tracking(
    tracked: dict[int, dict],
    next_id: int,
    validated: dict,
) -> tuple[dict[int, dict], int]:
    """Apply a validated model response to the tracking state.

    Removes lost objects, updates bounding boxes for still-tracked objects,
    and assigns new IDs to newly detected objects.

    Args:
        tracked: Current tracking state (modified in place).
        next_id: Next available tracking ID.
        validated: Validated response dict from validate_response.

    Returns:
        Tuple of (updated tracked dict, updated next_id).
    """
    for tid in validated["lost"]:
        tracked.pop(tid, None)

    for obj in validated["tracked"]:
        tid = obj["id"]
        if tid in tracked:
            tracked[tid]["bounding_box"] = obj["bounding_box"]

    for obj in validated["new"]:
        tracked[next_id] = {"label": obj["label"], "bounding_box": obj["bounding_box"]}
        next_id += 1

    return tracked, next_id


def draw_detections(img: np.ndarray, tracked: dict[int, dict]) -> np.ndarray:
    """Draw bounding boxes and tracking IDs on a copy of the image.

    Args:
        img: BGR image array.
        tracked: Current tracking state mapping ID to object dict.

    Returns:
        New image array with annotations drawn.
    """
    out = img.copy()
    for tid, obj in tracked.items():
        color = _BOX_COLORS[tid % len(_BOX_COLORS)]
        x1, y1, x2, y2 = (int(v) for v in obj["bounding_box"])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"[{tid}] {obj['label']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 5, color, -1)

    return out


def display_result(
    image_path: Path,
    response: str,
    elapsed_sec: float,
    tracked: dict[int, dict],
    parse_error: str,
) -> bool:
    """Display the annotated image and VLM response side by side in an OpenCV window.

    Bounding boxes for all currently tracked objects are drawn on the image.
    Press any key to advance to the next frame, or 'q' to quit.

    Args:
        image_path: Path to the source image.
        response: Raw model response text for the text panel.
        elapsed_sec: Inference time in seconds.
        tracked: Current tracking state after update, used to draw detections.
        parse_error: Non-empty string if JSON parsing failed (shown in panel).

    Returns:
        True to continue, False if the user pressed 'q'.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: could not read {image_path}")
        return True

    img = draw_detections(img, tracked)

    max_h = 600
    h, w = img.shape[:2]
    scale = max_h / h
    img = cv2.resize(img, (int(w * scale), max_h))
    h, w = img.shape[:2]

    panel_w = 700
    panel = np.full((h, panel_w, 3), 30, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 18
    x, y = 10, 24

    status = f"[{len(tracked)} tracked]" if not parse_error else "[INVALID JSON]"
    header_color = (100, 200, 255) if not parse_error else (60, 60, 255)
    for header in [f"Timestamp: {image_path.stem}", f"Inference:  {elapsed_sec:.2f}s  {status}", ""]:
        cv2.putText(panel, header, (x, y), font, 0.45, header_color, 1)
        y += line_h

    text_body = parse_error if parse_error else response
    chars_per_line = 72
    current = ""
    for word in text_body.split():
        if len(current) + len(word) + 1 > chars_per_line:
            cv2.putText(panel, current, (x, y), font, 0.45, (220, 220, 220), 1)
            y += line_h
            current = word
            if y > h - line_h:
                break
        else:
            current = (current + " " + word).strip()
    if current and y <= h - line_h:
        cv2.putText(panel, current, (x, y), font, 0.45, (220, 220, 220), 1)

    combined = np.hstack([img, panel])
    cv2.imshow("VLM Results  [any key = next | q = quit]", combined)
    return (cv2.waitKey(0) & 0xFF) != ord("q")


def run_inference(model: LLM, processor: AutoProcessor, image_paths: list[Path], prompt_template: str) -> None:
    """Run VLM inference over each image, maintaining object tracks across frames.

    Args:
        model: Loaded vLLM model.
        processor: The model's AutoProcessor.
        image_paths: Ordered list of image paths to process.
        prompt_template: Prompt string with a {tracked_objects} placeholder.
    """
    sampling_params = SamplingParams(max_tokens=1024)
    tracked: dict[int, dict] = {}
    next_id = 1

    for image_path in image_paths:
        with PilImage.open(image_path) as _img:
            img_w, img_h = _img.size
        inputs = build_inputs(image_path, prompt_template, processor, tracked)

        t_start = time.perf_counter()
        outputs = model.generate(inputs, sampling_params=sampling_params)
        elapsed = time.perf_counter() - t_start

        response = outputs[0].outputs[0].text
        print(f"[{image_path.stem}] ({elapsed:.2f}s) {response}")

        parse_error = ""
        parsed, parse_error = parse_response(response)
        if parsed is not None:
            validated, warnings = validate_response(parsed, tracked, img_w, img_h)
            for w in warnings:
                print(f"  [validation] {w}")
            tracked, next_id = update_tracking(tracked, next_id, validated)
        else:
            print(f"  [parse error] {parse_error}")

        if not display_result(image_path, response, elapsed, tracked, parse_error):
            break

    cv2.destroyAllWindows()


def main() -> None:
    """Load the dataset and model, then run VLM inference over each sampled frame."""
    home = Path.home()
    model_choice = Model.QWEN3_VL_8B_FP8
    checkpoint_path = home / "Qwen3-VL" / "models" / model_choice.value
    dataset_path = home / "data" / "Hercules_datasets" / "V2.4.C" / "data"
    prompt_path = home / "Qwen3-VL" / "PROMPT.md"
    robot_name = "Husky1"
    stride = 10

    prompt_template = load_prompt(prompt_path)
    image_paths = load_image_paths(dataset_path, robot_name, stride)
    model, processor = load_model(checkpoint_path)
    run_inference(model, processor, image_paths, prompt_template)


if __name__ == "__main__":
    main()
