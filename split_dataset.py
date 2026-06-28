import os, shutil, random

source_dir = "hand_gesture_dataset"   # your original dataset
train_dir = "dataset/train"
test_dir = "dataset/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for gesture in os.listdir(source_dir):
    gesture_path = os.path.join(source_dir, gesture)

    if not os.path.isdir(gesture_path):
        continue

    images = os.listdir(gesture_path)
    random.shuffle(images)

    split = int(0.8 * len(images))

    train_imgs = images[:split]
    test_imgs = images[split:]

    os.makedirs(os.path.join(train_dir, gesture), exist_ok=True)
    os.makedirs(os.path.join(test_dir, gesture), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(gesture_path, img),
                    os.path.join(train_dir, gesture, img))

    for img in test_imgs:
        shutil.copy(os.path.join(gesture_path, img),
                    os.path.join(test_dir, gesture, img))

print("✅ Dataset split completed!")