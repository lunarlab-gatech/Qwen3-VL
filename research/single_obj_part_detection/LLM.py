import sys
import json
import base64
import io
import math
import random
import time
import asyncio
from pathlib import Path

import cv2
import numpy as np
import openai
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part_detection"))
from prompt import parse_response, validate_response, parse_label_map, postprocess_detections

_SCRIPT_DIR  = Path(__file__).resolve().parent
_CLIENT      = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")
_MODEL       = "qwen3-vl"

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


async def infer(image_path: Path, prompt: str) -> tuple[str, object, float, str]:
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    img_info = f"{w}x{h} = {w*h:,} px"
    if w * h < 1_000_000:
        scale = math.sqrt(1_000_000 / (w * h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = pil_img.size
        img_info += f" → upscaled to {w}x{h} = {w*h:,} px"
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.perf_counter()
    completion = await _CLIENT.chat.completions.create(
        model=_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=4096,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - t0
    return completion.choices[0].message.content, completion.usage, elapsed, img_info


async def main():
    pairs = [
        (
            _SCRIPT_DIR / rel_image,
            (_SCRIPT_DIR / rel_prompt).read_text().strip(),
            rel_image,
        )
        for rel_image, rel_prompt in _IMAGE_PROMPT_MAP
    ]

    t_total = time.perf_counter()
    results = await asyncio.gather(*[infer(p, prompt) for p, prompt, _ in pairs])
    total_elapsed = time.perf_counter() - t_total

    all_boxes: dict[str, list] = {}
    per_prompt_metrics: list[dict] = []

    for (image_path, prompt, rel_image), (response, usage, elapsed, img_info) in zip(pairs, results):
        label_map = parse_label_map(prompt)
        print(f"[{rel_image}]")
        print(f"  [image]  {img_info}")
        print(f"  [timing] {elapsed:.1f}s | {usage.prompt_tokens} prompt tok | {usage.completion_tokens} completion tok | {usage.completion_tokens / elapsed:.1f} tok/s")
        print(f"  [response]\n{response}")

        parsed, parse_error = parse_response(response, label_map)
        if parsed is None:
            print(f"  [parse error] {parse_error}")
            detections = []
        else:
            detections, warnings = validate_response(parsed)
            for warning in warnings:
                print(f"  [validation] {warning}")
            detections = postprocess_detections(detections)

        all_boxes[str(image_path)] = detections
        per_prompt_metrics.append({
            "image": rel_image,
            "latency_s": round(elapsed, 3),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "completion_tok_per_s": round(usage.completion_tokens / elapsed, 2),
        })
        print()

    total_completion_tokens = sum(m["completion_tokens"] for m in per_prompt_metrics)
    total_prompt_tokens = sum(m["prompt_tokens"] for m in per_prompt_metrics)
    avg_latency = sum(m["latency_s"] for m in per_prompt_metrics) / len(per_prompt_metrics)
    avg_time_per_request = total_elapsed / len(per_prompt_metrics)
    throughput = total_completion_tokens / total_elapsed

    metrics = {
        "total_time_s": round(total_elapsed, 3),
        "avg_latency_s": round(avg_latency, 3),
        "avg_time_per_request_s": round(avg_time_per_request, 3),
        "overall_throughput_tok_per_s": round(throughput, 2),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "per_prompt": per_prompt_metrics,
    }

    print(f"[summary]")
    print(f"  total time        : {total_elapsed:.1f}s")
    print(f"  avg latency       : {avg_latency:.1f}s")
    print(f"  avg time/request  : {avg_time_per_request:.1f}s  (total / {len(per_prompt_metrics)} requests)")
    print(f"  overall throughput: {throughput:.1f} completion tok/s")
    print(f"  total tokens      : {total_prompt_tokens} prompt + {total_completion_tokens} completion\n")

    output_dir = _SCRIPT_DIR / "meronomy" / _MODEL
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(all_boxes, indent=2))
    print(f"Results saved to {results_path}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    for rel_image, _ in _IMAGE_PROMPT_MAP:
        image_path = str(_SCRIPT_DIR / rel_image)
        stem = Path(rel_image).stem
        draw_boxes(image_path, all_boxes, str(output_dir / f"visualized_{stem}.png"))


if __name__ == "__main__":
    asyncio.run(main())
