import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image

from model import ThreeLayerCNN
from dataset import get_transforms


def predict_single_image(image_path: str, model_path: str = "models/cat_dog_cnn3.pth", image_size: int = 128):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}. Train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture
    model = ThreeLayerCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Support state_dict or raw state dictionary
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()

    # Preprocess image
    _, eval_transform = get_transforms(image_size=image_size)
    img = Image.open(image_path).convert("RGB")
    tensor_input = eval_transform(img).unsqueeze(0).to(device)

    classes = ["Cat", "Dog"]

    with torch.no_grad():
        outputs = model(tensor_input)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, dim=0)

    predicted_label = classes[predicted_idx.item()]
    conf_pct = confidence.item() * 100

    print("-" * 40)
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_label} ({conf_pct:.2f}% confidence)")
    print(f"Probabilities -> Cat: {probs[0].item()*100:.2f}%, Dog: {probs[1].item()*100:.2f}%")
    print("-" * 40)

    return {
        "class": predicted_label,
        "confidence": conf_pct,
        "cat_prob": probs[0].item(),
        "dog_prob": probs[1].item()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog vs Cat Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="models/cat_dog_cnn3.pth", help="Path to saved .pth model file")
    parser.add_argument("--size", type=int, default=128, help="Image resolution for model input")
    args = parser.parse_args()

    predict_single_image(args.image, args.model, args.size)
