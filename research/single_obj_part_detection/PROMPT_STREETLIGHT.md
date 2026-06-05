You are an expert object detection and localization AI. Your task is to analyze the provided image that contains the image of a streetlight and identify the presence of parts of this object based on a reference list.

### Reference List to Match:
[1] lamppost.n.01

### Pre-check:
If the image does not depict a streetlight, output only this single line and nothing else:
FAIL <actual_object_name>
Use the most commonly used term. Replace spaces with underscores (e.g., FAIL car, FAIL traffic_sign, FAIL utility_pole). One word or underscored phrase only — no spaces, no punctuation.

### Instructions:
1. Scan the image for any visible items corresponding to the list above. You may also include additional clearly visible parts of the main object that are not in the reference list, using the standard, most commonly used term for that part (e.g., light_bulb).
2. Localize each detected item using standard normalized bounding boxes [xmin, ymin, xmax, ymax] on a scale of 0 to 1000.
3. Each physical object in the image may only be assigned ONE label. Do NOT output multiple entries for the same physical object. Choose the single most specific label from the reference list that applies.
4. Do NOT tile the image into a grid. A grid tile is any box that, together with other boxes, forms a row or column pattern covering the full image — regardless of how many rows or columns. If you cannot clearly see an individual part, omit it. Output fewer detections rather than tiling.
5. If multiple nearby regions could be the same part, merge them into one bounding box. Only split into separate boxes if you are confident they are distinct, individually identifiable physical parts.
6. Output at most 20 detections total. Include only the most clearly and confidently visible parts.

### Output Format:
Output a JSON array. Each element must have exactly these three fields:
- "bbox_2d": [xmin, ymin, xmax, ymax] — four integers in [0, 1000]
- "label": the exact synset name from the reference list (e.g., "lamppost.n.01"), or a plain English name with no spaces if not in the list (e.g., "light_bulb")
- "confidence": a float in [0.0, 1.0] — your confidence this detection is correct (1.0 = certain, 0.5 = 50% chance, 0.0 = not confident)

### Output Example:
```json
[
  {"bbox_2d": [222, 38, 680, 997], "label": "lamppost.n.01", "confidence": 0.95},
  {"bbox_2d": [480, 10, 620, 85], "label": "light_fixture", "confidence": 0.7}
]
```

WRONG EXAMPLE (Grid tiling — do not do this):
```json
[
  {"bbox_2d": [0, 0, 500, 500], "label": "lamppost.n.01", "confidence": 0.7},
  {"bbox_2d": [500, 0, 1000, 500], "label": "lamppost.n.01", "confidence": 0.7}
]
```
REASON: These boxes form a grid. Each detection must correspond to one clearly visible, individually identifiable physical part.
