import random
import cv2
import numpy as np
from PIL import Image

BOXES = {
    "/home/dbutterfield3/Downloads/1.jpg": [
        {"b": [386, 552, 595, 715], "l": "car_door.n.01"},
        {"b": [492, 725, 659, 814], "l": "car_wheel.n.01"},
        {"b": [504, 383, 715, 508], "l": "car_wheel.n.01"},
        {"b": [343, 563, 427, 672], "l": "car_window.n.01"},
        {"b": [350, 673, 420, 747], "l": "car_window.n.01"},
        {"b": [481, 269, 532, 408], "l": "window.n.02"},
        {"b": [390, 567, 427, 627], "l": "car_mirror.n.01"},
        {"b": [489, 90, 566, 262], "l": "grille.n.02"},
        {"b": [443, 94, 506, 381], "l": "hood.n.09"},
        {"b": [318, 388, 417, 596], "l": "roof.n.02"},
        {"b": [333, 335, 417, 574], "l": "windshield.n.01"},
    ],
    "/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_30/clipped_0.png": [
        {"b": [492, 0, 938, 172], "l": "car_wheel.n.01"},
        {"b": [453, 792, 853, 1000], "l": "car_wheel.n.01"},
        {"b": [308, 192, 755, 715], "l": "car_door.n.01"},
        {"b": [59, 131, 304, 529], "l": "car_window.n.01"}
    ],
    "/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_40/full_0.png": [
    {"b": [363, 244, 495, 466], "l": "license_plate"},
    {"b": [45, 142, 252, 794], "l": "rear_window.n.01"},
    {"b": [504, 786, 928, 893], "l": "wheel.n.01"},
    {"b": [257, 10, 679, 856], "l": "bumper.n.02"}
    ],
    "/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_43/full_0.png": [
    {"b": [70, 313, 915, 523], "l": "lamppost.n.01"}
    ]
}


def draw_boxes(image_path, output_path="meronomy/visualized_car.png"):
    pil_img = Image.open(image_path).convert("RGB")
    width, height = pil_img.size
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    scale = max(1, 800 // max(width, height))
    if scale > 1:
        img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        width, height = width * scale, height * scale

    if image_path not in BOXES:
        raise KeyError(f"No boxes entry for image path: {image_path!r}")

    for item in BOXES[image_path]:
        ymin, xmin, ymax, xmax = item["b"]
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


if __name__ == "__main__":
    draw_boxes("/home/dbutterfield3/Downloads/1.jpg", "meronomy/1.png")
    draw_boxes("/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_30/clipped_0.png", "meronomy/2.png")
    draw_boxes("/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_40/full_0.png", "meronomy/3.png")
    draw_boxes("/home/dbutterfield3/Research/ROMAN_DEVEL/research/Hercules/keyframe_images/Husky1/segment_43/full_0.png", "meronomy/4.png")