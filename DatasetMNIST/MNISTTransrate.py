import os
import struct
import numpy as np
from PIL import Image

def load_images(file_path):
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images(images, labels, subset, output_dir, resize_to=None, cmap=None):
    for i, (img, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(output_dir, subset, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        img_pil = Image.fromarray(img)

        # Optional: resize image
        if resize_to:
            img_pil = img_pil.resize(resize_to, Image.ANTIALIAS)

        # Optional: apply colormap (e.g., "hot", "jet")
        if cmap:
            img_pil = img_pil.convert("L").convert("RGB")  # to RGB for matplotlib cmap
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            cmap_func = cm.get_cmap(cmap)
            np_img = np.array(img_pil.convert("L")) / 255.0
            colored_img = (cmap_func(np_img)[:, :, :3] * 255).astype(np.uint8)
            img_pil = Image.fromarray(colored_img)

        img_path = os.path.join(label_dir, f"{i}.png")
        img_pil.save(img_path)

def main():
    # 修正されたファイルパス（Dockerコンテナ内）
    base_dir = "/workspace/DatasetMNIST/Originaldataset"
    files = {
        "train_images": os.path.join(base_dir, "train-images.idx3-ubyte"),
        "train_labels": os.path.join(base_dir, "train-labels.idx1-ubyte"),
        "test_images": os.path.join(base_dir, "t10k-images.idx3-ubyte"),
        "test_labels": os.path.join(base_dir, "t10k-labels.idx1-ubyte"),
    }

    output_dir = "/workspace/DatasetMNIST/MNIST_PNG"

    # Load data
    train_images = load_images(files["train_images"])
    train_labels = load_labels(files["train_labels"])
    test_images = load_images(files["test_images"])
    test_labels = load_labels(files["test_labels"])

    # Save as PNG
    save_images(train_images, train_labels, "train", output_dir,
                resize_to=(28, 28),
                cmap=None)

    save_images(test_images, test_labels, "test", output_dir,
                resize_to=(28, 28),
                cmap=None)

if __name__ == "__main__":
    main()
