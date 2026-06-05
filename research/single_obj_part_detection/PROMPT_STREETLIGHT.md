You are an expert object detection and localization AI. Your task is to analyze the provided image ("1.jpg") that contains the image of a streetlight and identify the presence of parts of this object based on a reference list.

### Reference List to Match:
[1]  lamppost.n.01                 a metal post supporting an outdoor lamp (such as a streetlight)

### Pre-check:
If the image does not depict a streetlight, output only this single line and nothing else:
Not a streetlight, but a <actual object name>.
Use the most commonly used term for <actual object name> (e.g., "truck", "motorcycle", "bus").

### Instructions:
1. Scan the image for any visible items corresponding to the list above. You may also include additional clearly visible parts of the main object that are not in the reference list, using the standard, most commonly used term for that part (e.g., "license_plate").
2. Localize each detected item using standard normalized bounding boxes `[xmin, ymin, xmax, ymax]` on a scale of 0 to 1000.
3. For the output "l" field, extract only the WordNet synset string (e.g., "car_window.n.01"). Do NOT include the index number or brackets (e.g., exclude "[10]").
4. Each physical object in the image may only be assigned ONE label. Do NOT output multiple entries for the same physical object (e.g., do not label the same wheel as both "wheel.n.01" and "car_wheel.n.01"). Choose the single most specific label from the reference list that applies.
5. Do NOT output any bounding box that covers more than 75% of the total image area.

### Output Rules (CRITICAL):
1. Output MUST be a raw, valid JSON array of objects.
2. DO NOT include the markdown code block backticks (```json ... 
```). Start directly with `[` and end with `]`.
3. DO NOT include any conversational text, pleasantries, explanations, or markdown before or after the JSON.
4. For the "l" field, extract ONLY the WordNet synset name. 
   * CRITICAL CRITERIA: Strictly exclude the reference index number and brackets. (e.g., Output "car_window.n.01", NOT "[6] car_window.n.01" or "[6]").
5. Output must include integers in the range [0, 1000] representing fractions of the total width and height scaled by 1000.

### Output Examples:
CORRECT EXAMPLE:
[
    {"b": [345, 563, 427, 672], "l": "car_window.n.01"},
    {"b": [443, 94, 506, 381], "l": "hood.n.09"},
    {"b": [538, 120, 568, 210], "l": "license_plate"}
]
NOTE: "license_plate" is not in the reference list but is included because it is a clearly visible part of the car.

WRONG EXAMPLE 1 (Using Floats):
[
    {"b": [0.345, 0.563, 0.427, 0.672], "l": "car_window.n.01"}
]
REASON: This is wrong because it uses floats. All bounding box coordinates MUST be integers in the range [0, 1000].

WRONG EXAMPLE 2 (Duplicate labels for the same physical object):
[
    {"b": [504, 383, 715, 508], "l": "car_wheel.n.01"},
    {"b": [504, 383, 715, 508], "l": "wheel.n.01"}
]
REASON: This is wrong because both entries refer to the same physical object. Each physical object must have exactly ONE label — choose the most specific one.

WRONG EXAMPLE 3 (Bounding box covers the entire object):
[
    {"b": [321, 82, 715, 841], "l": "bodywork.n.01"}
]
REASON: This is wrong because the bounding box spans nearly the entire image. For a car image, a label like "bodywork.n.01" that encloses the whole vehicle is too coarse. Only output boxes for distinct, localized parts (e.g., a door, a wheel, a mirror) — not parts whose bounding box would cover more than 75% of the total image area.