"""
Quick diagnostic: figures out which output index (0 or 1) the model
actually uses for "no tumor" vs "tumor", by running it against known
images from the dataset folder.

Run from inside backend/:
    python diagnose.py
"""
import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")

NO_TUMOR_DIR = os.path.join(BASE_DIR, "..", "dataset", "testing", "no_tumor")
TUMOR_DIR = os.path.join(BASE_DIR, "..", "dataset", "testing", "pituitary_tumor")

def load_and_predict(model, img_path):
    img = tf_image.load_img(img_path, target_size=(224, 224))
    arr = tf_image.img_to_array(img) 
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr, verbose=0)[0]

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model output shape:", model.output_shape)

    no_tumor_files = sorted(glob.glob(os.path.join(NO_TUMOR_DIR, "*")))[:10]
    tumor_files = sorted(glob.glob(os.path.join(TUMOR_DIR, "*")))[:10]

    print(f"\nTesting {len(no_tumor_files)} known NO-TUMOR images:")
    no_tumor_preds = []
    for f in no_tumor_files:
        pred = load_and_predict(model, f)
        no_tumor_preds.append(pred)
        print(f"  {os.path.basename(f):30s} -> raw={pred}  argmax={np.argmax(pred)}")

    print(f"\nTesting {len(tumor_files)} known TUMOR images:")
    tumor_preds = []
    for f in tumor_files:
        pred = load_and_predict(model, f)
        tumor_preds.append(pred)
        print(f"  {os.path.basename(f):30s} -> raw={pred}  argmax={np.argmax(pred)}")

    no_tumor_argmax = [int(np.argmax(p)) for p in no_tumor_preds]
    tumor_argmax = [int(np.argmax(p)) for p in tumor_preds]

    from collections import Counter
    no_tumor_mode = Counter(no_tumor_argmax).most_common(1)[0][0]
    tumor_mode = Counter(tumor_argmax).most_common(1)[0][0]

    print("\n" + "=" * 60)
    print(f"Most common argmax for NO-TUMOR images: {no_tumor_mode}")
    print(f"Most common argmax for TUMOR images:    {tumor_mode}")
    print("=" * 60)

    if no_tumor_mode == 0 and tumor_mode == 1:
        print("Model matches app.py's assumption (index 0=no, index 1=yes).")
        print("The class-order theory is NOT the cause -- something else is wrong.")
    elif no_tumor_mode == 1 and tumor_mode == 0:
        print("CONFIRMED: class order is REVERSED from what app.py assumes.")
        print("Fix: in predict_tumor(), swap the index 0/1 logic (see chat).")
    else:
        print("Inconclusive / mixed results -- model may not be reliably")
        print("distinguishing these classes at all (accuracy issue, not just ordering).")

if __name__ == "__main__":
    main()