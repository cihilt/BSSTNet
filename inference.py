from basicsr.archs.BSST_arch import BSST
import cv2
import numpy as np
import os
import torch.nn.functional as F


class SingleImageBSST(BSST):
    def forward(self, x):
        try:
            print(f"Processing patch of shape: {x.shape}")
            
            # Extract features
            feat = self.feat_extract(x)
            
            # Prepare dimensions for reconstruction
            b, c, h, w = feat.shape
            target_channels = 960
            repeats = target_channels // c
            
            # Add temporal dimension first
            feat = feat.unsqueeze(1)  # [B, T=1, C, H, W]
            feat = feat.repeat(1, 1, repeats, 1, 1)  # Repeat channels
            feat = feat.reshape(b, 1, target_channels, h, w)  # [B, T, C, H, W]
            
            # Reconstruction
            out = self.reconstruction(feat)
            
            # Extract final RGB image
            if out.dim() > 4:
                out = out.squeeze(1)
            out = out[:, :3]
            
            return x + out
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

def process_patch(model, patch):
    """Process a single patch."""
    with torch.no_grad():
        return model(patch)

def extract_patches(img, patch_size=256, overlap=48):
    """Extract overlapping patches from image."""
    _, c, h, w = img.shape
    patches = []
    positions = []
    
    stride = patch_size - overlap
    
    for y in range(0, h-overlap, stride):
        for x in range(0, w-overlap, stride):
            end_y = min(y + patch_size, h)
            end_x = min(x + patch_size, w)
            start_y = max(end_y - patch_size, 0)
            start_x = max(end_x - patch_size, 0)
            
            patch = img[:, :, start_y:end_y, start_x:end_x]
            patches.append(patch)
            positions.append((start_y, end_y, start_x, end_x))
            
    return patches, positions


def blend_patches(patches, positions, orig_shape, device='cuda:0'):
    """Blend processed patches back together."""
    b, c, h, w = orig_shape
    output = torch.zeros(orig_shape, device=device)
    weights = torch.zeros((h, w), device=device)
    
    for patch, (start_y, end_y, start_x, end_x) in zip(patches, positions):
        # Ensure patch is on correct device
        patch = patch.to(device)
        
        patch_h, patch_w = end_y - start_y, end_x - start_x
        weight = torch.ones((patch_h, patch_w), device=device)
        
        # Create blending weights
        if start_y > 0:
            weight[:32, :] *= torch.linspace(0, 1, 32, device=device)[:, None]
        if start_x > 0:
            weight[:, :32] *= torch.linspace(0, 1, 32, device=device)[None, :]
        if end_y < h:
            weight[-32:, :] *= torch.linspace(1, 0, 32, device=device)[:, None]
        if end_x < w:
            weight[:, -32:] *= torch.linspace(1, 0, 32, device=device)[None, :]
            
        # Apply weights
        output[:, :, start_y:end_y, start_x:end_x] += patch * weight[None, None, :, :]
        weights[start_y:end_y, start_x:end_x] += weight
    
    # Normalize
    output /= (weights + 1e-8)[None, None, :, :]
    return output

def deblur_image(model, input_path, output_path, patch_size=128, overlap=32):
    try:
        # Read and preprocess image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error reading image: {input_path}")
            return
            
        original_size = (img.shape[1], img.shape[0])
        print(f"Original image shape: {img.shape}")
        
        # Convert to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = img.unsqueeze(0)
        
        # Pad to be divisible by 32
        _, _, h, w = img.shape
        pad_h = (((h + 31) // 32) * 32) - h
        pad_w = (((w + 31) // 32) * 32) - w
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        print(f"Padded shape: {img.shape}")
        
        device = torch.device('cuda:0')
        # Move to GPU
        img = img.to(device)
        model = model.to(device)
        print(f"Using device: {device}")
        
        patches, positions = extract_patches(img, patch_size, overlap)
        print(f"Split into {len(patches)} patches")
        
        # Process each patch
        processed_patches = []
        for i, patch in enumerate(patches):
            print(f"Processing patch {i+1}/{len(patches)}")
            torch.cuda.empty_cache()  # Clear GPU memory
            output = process_patch(model, patch)
            # Keep on GPU
            processed_patches.append(output)
            
        # Blend patches
        output = blend_patches(processed_patches, positions, img.shape, device=device)
        
        # Remove padding if any
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]
        
        # Convert to image (only move to CPU at the final step)
        output = output.squeeze(0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, output)
        print(f"Saved deblurred image to: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        torch.cuda.empty_cache()

def main():
    print("Loading model...")
    try:
        # Load model
        model = SingleImageBSST()
        weights = torch.load('model_zoos/BSST_gopro.pth', map_location='cpu')
        
        if 'params_ema' in weights:
            model.load_state_dict(weights['params_ema'])
            print("Loaded EMA weights")
        else:
            model.load_state_dict(weights['params'])
            print("Loaded regular weights")
            
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Using GPU")
        else:
            print("Using CPU")
        
        # Process images
        input_dir = 'datasets/test_images'
        output_dir = 'results/deblurred'
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of images
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nFound {len(image_files)} images to process")
        for img_name in image_files:
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f'deblurred_{img_name}')
            print(f'\nProcessing {img_name}...')
            deblur_image(model, input_path, output_path)
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
