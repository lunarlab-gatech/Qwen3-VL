You are an expert object detection and localization AI. Your task is to analyze the provided image that contains the image of a car and identify the presence of parts of this object based on a reference list.

### Reference List to Match:
[1] drive_line.n.01, [2] glove_compartment.n.01, [3] automobile_horn.n.01, [4] rumble_seat.n.01, [5] automobile_engine.n.01, [6] sunroof.n.01, [7] high_gear.n.01, [8] axle.n.01, [9] luggage_compartment.n.01, [10] car_window.n.01, [11] suspension.n.05, [12] second_gear.n.01, [13] tailgate.n.01, [14] rear_window.n.01, [15] gearshift.n.01, [16] hood.n.09, [17] tail_fin.n.02, [18] pedal.n.02, [19] airbrake.n.02, [20] third_gear.n.01, [21] grille.n.02, [22] wheel.n.01, [23] power_brake.n.01, [24] brake_system.n.01, [25] fender.n.01, [26] car_door.n.01, [27] reverse.n.02, [28] cab.n.01, [29] chassis.n.03, [30] cockpit.n.03, [31] odometer.n.01, [32] cooling_system.n.01, [33] car_mirror.n.01, [34] floorboard.n.02, [35] car_seat.n.01, [36] window.n.02, [37] splashboard.n.02, [38] splasher.n.01, [39] speedometer.n.01, [40] windshield.n.01, [41] bodywork.n.01, [42] auto_accessory.n.01, [43] car_wheel.n.01, [44] gasoline_engine.n.01, [45] roof.n.02, [46] stabilizer_bar.n.01, [47] running_board.n.01, [48] brake.n.01, [49] buffer.n.06, [50] fuel_system.n.01, [51] electrical_system.n.02, [52] first_gear.n.01, [53] hand_brake.n.01, [54] bumper.n.02, [55] air_bag.n.01, [56] windshield_wiper.n.01, [57] accelerator.n.01, [58] internal-combustion_engine.n.01

### Pre-check:
If the image does not depict a car, output only this single line and nothing else:
FAIL <actual_object_name>
Use the most commonly used term. Replace spaces with underscores (e.g., FAIL truck, FAIL concrete_crack, FAIL stop_sign). One word or underscored phrase only — no spaces, no punctuation.

### Instructions:
1. Scan the image for any visible items corresponding to the list above. You may also include additional clearly visible parts of the main object that are not in the reference list, using the standard, most commonly used term for that part (e.g., license_plate).
2. Localize each detected item using standard normalized bounding boxes [xmin, ymin, xmax, ymax] on a scale of 0 to 1000.
3. Each physical object in the image may only be assigned ONE label. Do NOT output multiple entries for the same physical object. Choose the single most specific label from the reference list that applies.
4. Do NOT tile the image into a grid. A grid tile is any box that, together with other boxes, forms a row or column pattern covering the full image — regardless of how many rows or columns. If you cannot clearly see an individual part, omit it. Output fewer detections rather than tiling.
5. If multiple nearby regions could be the same part, merge them into one bounding box. Only split into separate boxes if you are confident they are distinct, individually identifiable physical parts.
6. Output at most 20 detections total. Include only the most clearly and confidently visible parts.

### Output Format:
Output a JSON array. Each element must have exactly these three fields:
- "bbox_2d": [xmin, ymin, xmax, ymax] — four integers in [0, 1000]
- "label": the exact synset name from the reference list (e.g., "car_mirror.n.01"), or a plain English name with no spaces if not in the list (e.g., "license_plate")
- "confidence": a float in [0.0, 1.0] — your confidence this detection is correct (1.0 = certain, 0.5 = 50% chance, 0.0 = not confident)

### Output Example:
```json
[
  {"bbox_2d": [150, 450, 280, 650], "label": "wheel.n.01", "confidence": 0.95},
  {"bbox_2d": [443, 94, 506, 381], "label": "hood.n.09", "confidence": 0.9},
  {"bbox_2d": [345, 563, 427, 672], "label": "car_window.n.01", "confidence": 0.75},
  {"bbox_2d": [538, 120, 568, 210], "label": "license_plate", "confidence": 0.6}
]
```

WRONG EXAMPLE (Grid tiling — do not do this):
```json
[
  {"bbox_2d": [0, 0, 500, 333], "label": "roof.n.02", "confidence": 0.7},
  {"bbox_2d": [500, 0, 1000, 333], "label": "roof.n.02", "confidence": 0.7},
  {"bbox_2d": [0, 333, 500, 666], "label": "car_door.n.01", "confidence": 0.7},
  {"bbox_2d": [500, 333, 1000, 666], "label": "car_door.n.01", "confidence": 0.7}
]
```
REASON: These boxes form a grid. Each detection must correspond to one clearly visible, individually identifiable physical part.
