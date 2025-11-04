import sys
import os

print("📁 Current working directory:", os.getcwd())
print("🧪 Testing package installation...")

try:
    import torch
    import torchvision
    import matplotlib
    import numpy as np
    print("✅ All packages imported successfully!")
    
    # Your ViT parameters
    seed = 14
    hidden_dim = 128 + (seed % 5) * 32
    num_heads = 4 + (seed % 3)
    patch_size = 8 + (seed % 4) * 2
    epochs = 10 + (seed % 5)
    
    print(f"\n🎯 ViT Parameters for Roll No. {seed}:")
    print(f"   Hidden Dimension: {hidden_dim}")
    print(f"   Number of Heads: {num_heads}")
    print(f"   Patch Size: {patch_size}")
    print(f"   Training Epochs: {epochs}")
    
except ImportError as e:
    print(f"Error: {e}")
    print("Run: pip install torch torchvision matplotlib numpy tqdm Pillow seaborn scikit-learn")