import albumentations as A
import numpy as np
import cv2
import os
import random
import glob
import shutil
from tqdm import tqdm

# =========================
# 1. Augmentation Functions
# =========================

def apply_night_effect(image, darkness_factor=0.15):
    """
    Darkens the image to simulate nighttime.
    """
    return (image * darkness_factor).astype(np.uint8)


def add_horizontal_oval_glow(image, center=None, rx=600, ry=250, strength=1.4):
    """
    Adds an oval glow horizontally on the image.
    center: (x, y) coordinates of the glow center. Defaults near bottom center.
    rx, ry: horizontal and vertical radii of the oval.
    strength: intensity multiplier of the glow.
    """
    h, w = image.shape[:2]

    if center is None:
        center = (w // 2, int(h * 0.65))

    Y, X = np.ogrid[:h, :w]
    dx = (X - center[0]) / rx
    dy = (Y - center[1]) / ry
    dist = np.sqrt(dx*dx + dy*dy)

    mask = np.clip(1 - dist, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rx / 10)[..., None]

    result = image.astype(np.float32)
    result *= (1 + mask * (strength - 1))

    return np.clip(result, 0, 255).astype(np.uint8)


# Rain augmentation using Albumentations
rain_aug = A.Compose([
    A.RandomRain(
        brightness_coefficient=0.9,  # slightly darken the image
        slant_range=(-45, 45),
        drop_width=1, drop_length=50,
        blur_value=10,
        rain_type="heavy",
        p=1.0
    )
])

def augment_image(image):
    """
    Apply random augmentations: night effect, rain effect, or both.
    Guarantees at least one augmentation per image.
    """
    augmented = image.copy()

    # Randomly decide which augmentations to apply
    apply_night = random.random() < 0.5
    apply_rain = random.random() < 0.5

    # Ensure at least one augmentation is applied
    if not (apply_night or apply_rain):
        if random.random() < 0.5:
            apply_night = True
        else:
            apply_rain = True

    # Apply night effect
    if apply_night:
        augmented = apply_night_effect(augmented)
        augmented = add_horizontal_oval_glow(augmented, rx=1000, ry=120, strength=20)

    # Apply rain effect
    if apply_rain:
        augmented = rain_aug(image=augmented)["image"]

    return augmented

# =========================
# 2. Dataset Expansion
# =========================

images_folder = "Augmentation_01\\data_null_30\\test\\images"
labels_folder = "Augmentation_01\\data_null_30\\test\\labels"

# Get all image paths in the folder
image_paths = glob.glob(os.path.join(images_folder, "*.*"))  # adjust pattern if needed
num_to_augment = len(image_paths) // 3  # augment ~33% of images

# Randomly select images for augmentation
images_to_augment = random.sample(image_paths, num_to_augment)

count = 0

# Loop through selected images
for img_path in tqdm(images_to_augment, desc="Augmenting images"):
    # Load image
    img = cv2.imread(img_path)
    augmented_img = augment_image(img)

    # Prepare new image filename
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    new_image_name = f"aug_{name}{ext}"
    new_image_path = os.path.join(images_folder, new_image_name)

    # Save augmented image
    cv2.imwrite(new_image_path, augmented_img)

    # --- Copy corresponding label file ---
    label_path = os.path.join(labels_folder, f"{name}.txt")
    if os.path.exists(label_path):
        new_label_name = f"aug_{name}.txt"
        new_label_path = os.path.join(labels_folder, new_label_name)
        shutil.copy(label_path, new_label_path)
    else:
        print(f"⚠ No label found for: {name}.txt")

    count += 1

# =========================
# 3. Summary
# =========================
print(f"\nOriginal images: {len(image_paths)}")
print(f"Augmented images saved: {count}")
print(f"Total images after augmentation: {len(image_paths) + count}")