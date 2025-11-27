
import os
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# Clear CUDA cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Resolve paths relative to this file
HERE = os.path.dirname(__file__)

# Load class names
CLASSES_PATH = os.path.abspath(os.path.join(HERE, 'classes.json'))
with open(CLASSES_PATH) as f:
    class_names = json.load(f)

# Device - force CPU to save memory
device = torch.device("cpu")

# Initialize model variable
model = None

def load_model():
    """Load the model only when needed to save memory"""
    global model
    if model is not None:
        return model
        
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model with minimal memory footprint
    with torch.no_grad():
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(class_names))
        MODEL_PATH = os.path.abspath(os.path.join(HERE, '..', 'models', 'vit_brain_tumor.pth'))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prediction function
def predict_image(image_path):
    """Make prediction with memory optimization"""
    # Load model only when needed
    model = load_model()
    
    image = Image.open(image_path).convert("RGB")
    imgs = [image, image.transpose(Image.FLIP_LEFT_RIGHT)]  # Test-time augmentation
    
    with torch.no_grad():
        model.eval()
        prob_accum = None
        
        for im in imgs:
            img_tensor = transform(im).unsqueeze(0).to(device)
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            prob_accum = probs if prob_accum is None else prob_accum + probs
            
            # Clear memory after each prediction
            del img_tensor, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        prob_accum /= len(imgs)
        pred_idx = torch.argmax(prob_accum).item()
        confidence = prob_accum[0][pred_idx].item()
        
    return pred_idx, confidence, prob_accum[0].cpu().numpy()

# Grad-CAM - only initialize when needed
gradcam = None

def get_gradcam():
    """Lazy initialization of Grad-CAM to save memory"""
    global gradcam
    if gradcam is None:
        model = load_model()
        target_layer = model.conv_proj
        gradcam = LayerGradCam(model, target_layer)
    return gradcam

def generate_gradcam(image_path, output_dir=None, save_path=None):
    """Generate Grad-CAM visualization with memory optimization"""
    # Load model and get gradcam only when needed
    model = load_model()
    gradcam = get_gradcam()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
        
        # Generate Grad-CAM
        model.zero_grad()
        attributions = gradcam.attribute(input_tensor, target=pred_class)
        cam = attributions[0].mean(0).cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Clean up
        del input_tensor, output, attributions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return cam, pred_class
        
    except Exception as e:
        print(f"Error in generate_gradcam: {str(e)}")
        raise
        # Resize CAM to match input image size
        cam = cam.reshape(14, 14)
        cam_resized = np.array(Image.fromarray(cam).resize(image.size, resample=Image.BILINEAR))

        # --- Brain foreground mask to keep hotspot inside the head ---
        # Build a coarse brain mask from grayscale intensity (MRI background is usually dark)
        gray = np.asarray(image.convert('L')).astype('float32') / 255.0
        thr_fg = max(0.20, float(np.quantile(gray, 0.35)))
        brain_mask = (gray > thr_fg).astype(np.uint8)
        H, W = brain_mask.shape
        
        # Apply mask to CAM
        cam_resized = cam_resized * brain_mask
        
        # Clean up more memory
        del gray, brain_mask
        
        return cam_resized, pred_class
    def dilate3x3(m):
        pad = 1
        mp = np.pad(m, pad, mode='constant')
        return (
            (mp[0:H,   0:W] | mp[0:H,   1:W+1] | mp[0:H,   2:W+2] |
             mp[1:H+1, 0:W] | mp[1:H+1, 1:W+1] | mp[1:H+1, 2:W+2] |
             mp[2:H+2, 0:W] | mp[2:H+2, 1:W+1] | mp[2:H+2, 2:W+2])
        ).astype(np.uint8)
    def erode3x3(m):
        pad = 1
        mp = np.pad(m, pad, mode='constant')
        return (
            (mp[0:H,   0:W] & mp[0:H,   1:W+1] & mp[0:H,   2:W+2] &
             mp[1:H+1, 0:W] & mp[1:H+1, 1:W+1] & mp[1:H+1, 2:W+2] &
             mp[2:H+2, 0:W] & mp[2:H+2, 1:W+1] & mp[2:H+2, 2:W+2])
        ).astype(np.uint8)
    brain_mask = dilate3x3(brain_mask)
    brain_mask = dilate3x3(brain_mask)
    brain_mask = erode3x3(brain_mask)
    brain_mask = erode3x3(brain_mask)
    brain_mask = erode3x3(brain_mask)
    # Remove a small border margin to avoid picking outside cranial vault
    margin = max(2, int(0.03 * min(H, W)))
    brain_mask[:margin, :] = 0
    brain_mask[-margin:, :] = 0
    brain_mask[:, :margin] = 0
    brain_mask[:, -margin:] = 0
    # Threshold to create a binary mask (use a higher quantile to focus on peak tumor region)
    thr = float(np.quantile(cam_resized, 0.95))
    mask = (cam_resized >= thr).astype(np.uint8)  # 0/1
    # Enforce brain interior constraint early
    mask = (mask & brain_mask).astype(np.uint8)

    # Dimensions and coordinate grids
    h, w = cam_resized.shape
    yy, xx = np.ogrid[:h, :w]

    # Light dilation (3x3) to make the region a bit fuller and contiguous
    if mask.any():
        pad = 1
        mp = np.pad(mask, pad, mode='constant')
        # 3x3 max filter
        mask_dil = (
            (mp[0:h,     0:w]   | mp[0:h,     1:w+1] | mp[0:h,     2:w+2] |
             mp[1:h+1,   0:w]   | mp[1:h+1,   1:w+1] | mp[1:h+1,   2:w+2] |
             mp[2:h+2,   0:w]   | mp[2:h+2,   1:w+1] | mp[2:h+2,   2:w+2])
        ).astype(np.uint8)
        mask = mask_dil

    # Keep only the connected component that contains the peak activation (inside the brain)
    if mask.any():
        cam_for_peak = cam_resized * brain_mask
        if cam_for_peak.max() > 0:
            seed_y, seed_x = np.unravel_index(np.argmax(cam_for_peak), cam_for_peak.shape)
        else:
            seed_y, seed_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
        visited = np.zeros_like(mask, dtype=np.uint8)
        if mask[seed_y, seed_x] == 0:
            # If peak is just below threshold, nudge by seeding closest on-mask pixel
            ys_, xs_ = np.where(mask > 0)
            if ys_.size:
                # choose nearest mask pixel to peak
                d2 = (ys_ - seed_y)**2 + (xs_ - seed_x)**2
                i = int(np.argmin(d2))
                seed_y, seed_x = int(ys_[i]), int(xs_[i])
        stack = [(int(seed_y), int(seed_x))] if mask[seed_y, seed_x] == 1 else []
        while stack:
            y0, x0 = stack.pop()
            if visited[y0, x0] == 1:
                continue
            visited[y0, x0] = 1
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                y1, x1 = y0+dy, x0+dx
                if 0 <= y1 < h and 0 <= x1 < w and mask[y1, x1] == 1 and visited[y1, x1] == 0:
                    stack.append((y1, x1))
        mask = visited

    # Compute peak inside brain for a circular/round hotspot
    cam_for_peak = cam_resized * brain_mask
    if cam_for_peak.max() > 0:
        cy, cx = np.unravel_index(np.argmax(cam_for_peak), cam_for_peak.shape)
    else:
        cy, cx = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

    # Determine radius from mask bbox if available to size the circle sensibly
    ys, xs = np.where(mask > 0)
    bbox = None
    if ys.size > 0 and xs.size > 0:
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        bbox = (x1, y1, x2, y2)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        est_r = int(0.8 * max(x2 - x1, y2 - y1) / 2.0)
    else:
        est_r = int(0.25 * min(h, w))
    sigma = max(est_r, 8)

    # Build radial Gaussian blob centered at (cx, cy)
    dy = yy - cy
    dx = xx - cx
    d2 = (dy*dy + dx*dx).astype('float32')
    gaussian = np.exp(-d2 / (2.0 * float(sigma*sigma)))
    # Use gaussian both as heat value and alpha driver, clipped to brain
    heat_val = gaussian
    alpha = (gaussian * brain_mask.astype('float32'))
    # Normalize alpha to [0,1] and smooth
    alpha = np.clip(alpha / (alpha.max() + 1e-6), 0.0, 1.0)
    if alpha.any():
        k = 5
        pad = k // 2
        a_p = np.pad(alpha, pad, mode='edge')
        alpha = (
            a_p[0:h,     0:w]   + a_p[0:h,     1:w+1] + a_p[0:h,     2:w+2] +
            a_p[1:h+1,   0:w]   + a_p[1:h+1,   1:w+1] + a_p[1:h+1,   2:w+2] +
            a_p[2:h+2,   0:w]   + a_p[2:h+2,   1:w+1] + a_p[2:h+2,   2:w+2]
        ) / 9.0
        alpha = np.clip(alpha, 0.0, 1.0).astype('float32')

    # Build jet colors from the gaussian heat value
    base = np.array(image).astype('float32')
    cmap_rgb = (plt.get_cmap('jet')(np.clip(heat_val, 0.0, 1.0))[..., :3] * 255).astype('float32')
    alpha3 = np.stack([alpha, alpha, alpha], axis=-1)
    # Subtle transparent yellowish background tint beneath the hotspot
    bg_alpha = 0.18
    yellow = np.zeros_like(base)
    yellow[..., 0] = 255.0
    yellow[..., 1] = 255.0
    bg_tinted = base * (1.0 - bg_alpha) + yellow * bg_alpha
    # Overlay hotspot onto yellow-tinted base (location and size unchanged)
    overlay_arr = bg_tinted * (1.0 - alpha3) + cmap_rgb * alpha3
    overlay_arr = np.clip(overlay_arr, 0, 255).astype('uint8')
    overlay_img = Image.fromarray(overlay_arr)

    # Determine output path
    if save_path is None:
        if output_dir is None:
            output_dir = HERE
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"gradcam_{base}.png")

    # Save refined overlay with bbox
    overlay_img.save(save_path)

    # Also save mask image for debugging/usage
    mask_img = Image.fromarray(np.uint8(mask*255), mode='L')
    mask_path = os.path.join(output_dir, f"gradcam_mask_{base}.png")
    mask_img.save(mask_path)

    return {
        "overlay_path": save_path,
        "mask_path": mask_path,
        "bbox": bbox,
        "threshold": thr,
        "center": (int(cx), int(cy)),
    }
