import matplotlib
matplotlib.use('Agg')

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import joblib

from model import CSRNet

# ========================
#  Load Deep Learning model
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading CSRNet (deep) on {device}...')
deep_model = CSRNet(pretrained=False).to(device)
deep_model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
deep_model.eval()
print('CSRNet loaded.')

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========================
#  Load Traditional CV model
# ========================
print('Loading traditional model...')
traditional_model = joblib.load('traditional_model.pkl')
print('Traditional model loaded.')


# ========================
#  Prediction functions
# ========================
def predict_deep(img):
    """CSRNet deep learning prediction."""
    if img is None:
        return 0, None

    target_w, target_h = 640, 480
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        density = deep_model(img_tensor)
        count = density.sum().item()

    density_np = density.cpu().squeeze().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img)
    ax1.set_title('Input (Deep)', fontsize=12)
    ax1.axis('off')
    ax2.imshow(density_np, cmap='jet')
    ax2.set_title('Density Map (Deep)', fontsize=12)
    ax2.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return int(round(count)), Image.open(buf)


def predict_traditional(img):
    """Traditional CV (GBR + hand-crafted features) prediction."""
    if img is None:
        return 0, None

    target_w, target_h = 640, 480
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)

    # Global count prediction
    count = traditional_model.predict(img_resized)

    # Patch-based density map (coarse 10x13 grid for speed)
    density_map, _ = traditional_model.predict_density_map(
        img_resized, grid_h=10, grid_w=13
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img)
    ax1.set_title('Input (Traditional)', fontsize=12)
    ax1.axis('off')
    ax2.imshow(density_map, cmap='jet', interpolation='bilinear')
    ax2.set_title('Density Map (Traditional)', fontsize=12)
    ax2.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return int(round(count)), Image.open(buf)


# ========================
#  Gradio Blocks UI
# ========================
with gr.Blocks(title='Crowd Counting — Deep vs Traditional') as demo:
    gr.Markdown(
        '# Crowd Counting: Deep Learning vs Traditional CV\n'
        'Upload an image to compare CSRNet (deep) with GBR + hand-crafted features (traditional).'
    )

    with gr.Row():
        img_input = gr.Image(type='pil', label='Upload Image', scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown('## Deep Learning (CSRNet v4)')
            deep_count = gr.Number(label='Predicted Count')
            deep_density = gr.Image(type='pil', label='Density Map')

        with gr.Column():
            gr.Markdown('## Traditional CV (GBR + Hand-crafted Features)')
            trad_count = gr.Number(label='Predicted Count')
            trad_density = gr.Image(type='pil', label='Density Map')

    # Trigger prediction on image upload
    img_input.change(
        fn=predict_deep, inputs=img_input, outputs=[deep_count, deep_density]
    )
    img_input.change(
        fn=predict_traditional, inputs=img_input, outputs=[trad_count, trad_density]
    )

if __name__ == '__main__':
    demo.launch(theme='soft', inbrowser=True)
