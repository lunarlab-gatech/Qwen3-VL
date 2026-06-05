import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reuse model/inference utilities from part_detection
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part_detection"))
from prompt import load_model, build_inputs, parse_response, validate_response, Model

_SCRIPT_DIR = Path(__file__).resolve().parent

_IMAGE_PROMPT_MAP: list[tuple[str, str]] = [
    ("images/1.jpg", "PROMPT_CAR.md"),
    ("images/2.png", "PROMPT_CAR.md"),
    ("images/3.png", "PROMPT_CAR.md"),
    ("images/4.png", "PROMPT_STREETLIGHT.md"),
    ("images/5.png", "PROMPT_CAR.md"),
]


def draw_boxes(image_path, boxes, output_path="meronomy/visualized_car.png"):
    pil_img = Image.open(image_path).convert("RGB")
    width, height = pil_img.size
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    scale = max(1, 800 // max(width, height))
    if scale > 1:
        img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        width, height = width * scale, height * scale

    if image_path not in boxes:
        raise KeyError(f"No boxes entry for image path: {image_path!r}")

    for item in boxes[image_path]:
        xmin, ymin, xmax, ymax = item["b"]
        label = item["l"]
        cv2_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        start_point = (int(xmin / 1000 * width), int(ymin / 1000 * height))
        end_point = (int(xmax / 1000 * width), int(ymax / 1000 * height))

        cv2.rectangle(img, start_point, end_point, cv2_color, 3)

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_end_point = (start_point[0] + text_size[0] + 10, start_point[1] - text_size[1] - 10)

        if start_point[1] - text_size[1] - 10 < 0:
            text_end_point = (start_point[0] + text_size[0] + 10, start_point[1] + text_size[1] + 10)
            cv2.rectangle(img, start_point, text_end_point, cv2_color, -1)
            tx, ty = start_point[0] + 5, start_point[1] + text_size[1] + 5
        else:
            cv2.rectangle(img, (start_point[0], start_point[1] - text_size[1] - 10), text_end_point, cv2_color, -1)
            tx, ty = start_point[0] + 5, start_point[1] - 5

        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")


def main():
    checkpoint_path = Path.home() / "Qwen3-VL" / "models" / Model.QWEN3_VL_8B_FP8.value
    model, processor = load_model(checkpoint_path)

    all_boxes: dict[str, list] = {}

    for rel_image, rel_prompt in _IMAGE_PROMPT_MAP:
        image_path = _SCRIPT_DIR / rel_image
        prompt = (_SCRIPT_DIR / rel_prompt).read_text().strip()

        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        inputs = build_inputs(image_rgb, prompt, processor)

        from vllm import SamplingParams
        outputs = model.generate(inputs, SamplingParams(max_tokens=1024))
        response = outputs[0].outputs[0].text
        print(f"[{rel_image}] {response}")

        parsed, parse_error = parse_response(response)
        if parsed is None:
            print(f"  [parse error] {parse_error}")
            detections = []
        else:
            detections, warnings = validate_response(parsed)
            for w in warnings:
                print(f"  [validation] {w}")

        all_boxes[str(image_path)] = detections

    output_dir = _SCRIPT_DIR / "meronomy"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(
        {k: v for k, v in all_boxes.items()}, indent=2
    ))
    print(f"Results saved to {results_path}")

    for rel_image, _ in _IMAGE_PROMPT_MAP:
        image_path = str(_SCRIPT_DIR / rel_image)
        if all_boxes.get(image_path):
            stem = Path(rel_image).stem
            draw_boxes(image_path, all_boxes, str(output_dir / f"visualized_{stem}.png"))


if __name__ == "__main__":
    main()
