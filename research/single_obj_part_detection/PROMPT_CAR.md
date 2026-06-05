You are an expert object detection and localization AI. Your task is to analyze the provided image that contains the image of a car and identify the presence of parts of this object based on a reference list.

### Reference List to Match:
[1] drive_line.n.01, [2] glove_compartment.n.01, [3] automobile_horn.n.01, [4] rumble_seat.n.01, [5] automobile_engine.n.01, [6] sunroof.n.01, [7] high_gear.n.01, [8] axle.n.01, [9] luggage_compartment.n.01, [10] car_window.n.01, [11] suspension.n.05, [12] second_gear.n.01, [13] tailgate.n.01, [14] rear_window.n.01, [15] gearshift.n.01, [16] hood.n.09, [17] tail_fin.n.02, [18] pedal.n.02, [19] airbrake.n.02, [20] third_gear.n.01, [21] grille.n.02, [22] wheel.n.01, [23] power_brake.n.01, [24] brake_system.n.01, [25] fender.n.01, [26] car_door.n.01, [27] reverse.n.02, [28] cab.n.01, [29] chassis.n.03, [30] cockpit.n.03, [31] odometer.n.01, [32] cooling_system.n.01, [33] car_mirror.n.01, [34] floorboard.n.02, [35] car_seat.n.01, [36] window.n.02, [37] splashboard.n.02, [38] splasher.n.01, [39] speedometer.n.01, [40] windshield.n.01, [41] bodywork.n.01, [42] auto_accessory.n.01, [43] car_wheel.n.01, [44] gasoline_engine.n.01, [45] roof.n.02, [46] stabilizer_bar.n.01, [47] running_board.n.01, [48] brake.n.01, [49] buffer.n.06, [50] fuel_system.n.01, [51] electrical_system.n.02, [52] first_gear.n.01, [53] hand_brake.n.01, [54] bumper.n.02, [55] air_bag.n.01, [56] windshield_wiper.n.01, [57] accelerator.n.01, [58] internal-combustion_engine.n.01

### Pre-check:
If the image does not depict a car, output only this single line and nothing else:
Not a car, but a <actual object name>.
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