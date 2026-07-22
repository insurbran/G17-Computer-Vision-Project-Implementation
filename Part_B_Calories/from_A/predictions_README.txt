predictions.json - how to read it (Track A -> Tracks B and C)
=============================================================

This file is the output of the Track A food detector. It lists, for each image,
the food items found and where they are. Tracks B (calories) and C (demo) both
read from this.

STRUCTURE
---------
{
  "schema_version": 1,
  "model": "best.pt",
  "run": "baseline3",
  "conf_threshold": 0.25,
  "iou_threshold": 0.70,
  "classes": ["Anchovies", "Boiled-Egg", ... 17 total],
  "predictions": [
    {
      "image_id": "somephoto.jpg",
      "image_size": [width, height],
      "boxes": [
        {
          "class": "Fried-Rice",
          "class_id": 7,
          "conf": 0.9124,
          "xyxy": [120.5, 88.0, 890.2, 720.4]
        }
      ]
    }
  ]
}

WHAT YOU NEED TO KNOW
---------------------
1. xyxy is [left, top, right, bottom] in absolute pixels, in the ORIGINAL image
   size (see image_size), not 640x640. Crop straight from the source photo, no
   rescaling needed. This is the usual place an integration breaks, so test it
   early with one image.

2. boxes is sorted by confidence, highest first.

3. boxes can be empty for an image (none of the current 76 are, but handle the
   empty case anyway so your code does not crash on it).

4. Coordinates are floats. Round them yourself if you need integer pixel indices.

5. There are NO calories in this file, on purpose. The detector reports what food
   is where and how confident it is. Turning that into calories is Track B's job.

6. class_id is the index into the "classes" list. class and class_id always match,
   use whichever is easier for your code.

SCOPE
-----
- The detector only knows the 17 classes in the "classes" list. Anything outside
  those (for example laksa, or non-food) it cannot detect.

- These predictions are for the 76 test images only. If you want the demo to run
  on a fresh photo, that needs the trained model, not this file. Ask me and I will
  send the weights (best.pt) plus a short snippet showing how to call the model on
  any image. This file alone cannot do new photos.

QUESTIONS
---------
Anything unclear or anything breaks, message me before building around it.
