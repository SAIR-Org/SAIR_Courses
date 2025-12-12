import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

os.makedirs("samples/mnist", exist_ok=True)
os.makedirs("samples/fashion", exist_ok=True)
os.makedirs("samples/cifar10", exist_ok=True)

# -------------------------------
# 1) MNIST Samples
# -------------------------------
mnist = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True
)

saved = set()
for img, label in mnist:
    img.save(f"samples/mnist/{label}.png")
    saved.add(label)
    if len(saved) == 10:
        break

print("MNIST samples saved → samples/mnist/*")

# -------------------------------
# 2) Fashion-MNIST Samples (fixed)
# -------------------------------
fashion = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True
)

fashion_names = [
    "T-shirt_top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]

saved = set()
for img, label in fashion:
    safe_name = fashion_names[label].replace("/", "_")
    img.save(f"samples/fashion/{safe_name}.png")
    saved.add(label)
    if len(saved) == 10:
        break

print("Fashion-MNIST samples saved → samples/fashion/*")

# -------------------------------
# 3) CIFAR-10 Samples
# -------------------------------
cifar = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

cifar_names = [name.replace("/", "_") for name in cifar.classes]

saved = set()
for img, label in cifar:
    img.save(f"samples/cifar10/{cifar_names[label]}.png")
    saved.add(label)
    if len(saved) == 10:
        break

print("CIFAR-10 samples saved → samples/cifar10/*")
