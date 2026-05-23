import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from model import CSRNet


def predict(model, img_path, device):
    """Predict the number of people in a single image."""
    model.eval()

    img = Image.open(img_path).convert('RGB')

    target_w, target_h = 640, 480
    img = img.resize((target_w, target_h), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        density = model(img_tensor)
        count = density.sum().item()

    return round(count), density.cpu().squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description='Crowd Counting Inference')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model weights')
    parser.add_argument('--show-density', action='store_true', help='Show density map')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CSRNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    count, density = predict(model, args.image, device)

    print(f'Predicted count: {count}')

    if args.show_density:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        img = Image.open(args.image)
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(density, cmap='jet')
        ax2.set_title(f'Density Map (count={count})')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
