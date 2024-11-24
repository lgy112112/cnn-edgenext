import torch
from torch import nn
from .sdta_encoder import SDTAEncoder

class SimpleCNNWithSDTA(nn.Module):
    def __init__(self, in_chans=3, num_classes=10, dim=64, num_heads=4, scales=1, 
                 drop_path=0.0, layer_scale_init_value=1e-6, expan_ratio=4):
        super(SimpleCNNWithSDTA, self).__init__()
        
        # Convolutional Encoder Layer
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=7, stride=2, padding=3),  # Basic convolution layer with 7x7 kernel
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        
        # SDTA Encoder Layer
        self.sdta_encoder = SDTAEncoder(
            dim=dim, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
            expan_ratio=expan_ratio, use_pos_emb=True, num_heads=num_heads, scales=scales
        )
        
        # Classification Head
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # Convolutional Encoder Forward
        x = self.conv_encoder(x)  # (N, C, H, W)
        
        # SDTA Encoder Forward
        x = self.sdta_encoder(x)  # (N, C, H, W)
        
        # Global Average Pooling and Fully Connected Layer for Classification
        B, C, H, W = x.shape
        x = x.mean([-2, -1])  # Global average pooling over (H, W)
        x = self.norm(x)  # (N, C)
        x = self.head(x)  # (N, num_classes)
        
        return x

if __name__ == "__main__":
    # Define the model
    model = SimpleCNNWithSDTA(in_chans=3, num_classes=10, dim=64, num_heads=4, scales=2)
    
    # Create dummy input tensor with batch size 2 and 3 channels (RGB image of size 64x64)
    dummy_input = torch.randn(2, 3, 227, 227)
    
    # Run dummy input through the model
    output = model(dummy_input)
    
    # Print output shape
    print(f"Output shape: {output.shape}")
    # Expected output shape: (2, 10), where 2 is the batch size and 10 is the number of classes
