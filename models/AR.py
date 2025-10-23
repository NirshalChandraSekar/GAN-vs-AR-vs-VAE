import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"

        self.register_buffer("mask", self.weight.data.clone())

        ############################
        # TODO: Fill the mask.
        # - Type A: exclude the center pixel (first layer).
        # - Type B: include the center pixel (subsequent layers).
        # Make sure to zero out all “future” pixels in raster order.
        ############################

        kh, kw = self.kernel_size
        yc, xc = kh // 2, kw // 2 # center position

        self.mask.fill_(1)
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc+1:] = 0

        if mask_type == 'A':
            self.mask[:, :, yc, xc] = 0

    def forward(self, x):
        ############################
        # TODO: Multiply the weight tensor (self.weight) by the mask (self.mask) before convolution.
        # Then perform normal convolution.
        ############################
        masked_weight = self.weight * self.mask
        
        out = F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()

        ############################
        # TODO: Implement unconditional PixelCNN
        # - First layer: MaskedConv2d with type 'A'
        # - Remaining layers: MaskedConv2d with type 'B'
        # - Use nonlinear activations (e.g. ReLU) after each masked conv
        # - You may also add normalization (BatchNorm) if desired between conv and ReLU
        # - You are also welcome to explore residual connections as original paper but not required to do so
        ############################

        self.model = nn.Sequential(
            MaskedConv2d('A', 1, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            MaskedConv2d('B', 64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            MaskedConv2d('B', 64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            MaskedConv2d('B', 64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            MaskedConv2d('B', 64, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=1)
        )

        ############################
        # TODO: Final output layer
        # - 1x1 convolution mapping from hidden channels -> 1 channel (for binary MNIST pixels)
        ############################

        # Already included in self.model above


    def forward(self, x):
        """
        x: (N, 1, H, W) input images
        """
        ############################
        # TODO: Forward pass through network
        ############################
        out = self.model(x)
        return out

class ConditionalMaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, num_classes,
                 stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        
        self.register_buffer("mask", self.weight.data.clone())
        
        ############################
        # TODO: Construct the masks
        # (same as in MaskedConv2d)
        ############################

        kh, kw = self.kernel_size
        yc, xc = kh // 2, kw // 2 # center position

        self.mask.fill_(1)
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc+1:] = 0

        if mask_type == 'A':
            self.mask[:, :, yc, xc] = 0

        ############################
        # TODO: Add class-conditioning
        # - Create a learnable projection (V) from one-hot class labels into out_channels
        # - Store it as self.class_emb (e.g. a Linear layer)
        ############################
        self.class_emb = nn.Linear(num_classes, out_channels)

    def forward(self, x, y):
        ############################
        # TODO: Multiply the weight tensor (self.weight) by the mask (self.mask) before convolution. Then perform normal convolution.
        # This is same as MaskedConv2d
        ############################
        masked_weight = self.weight * self.mask
        out = F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        ############################
        # TODO: Compute class-dependent bias = V*y --> Shape: (N, out_channels)
        ############################
        cond_bias = self.class_emb(y)
        out = out + cond_bias[:, :, None, None]  # Broadcast spatially and add to convolution output
        return out

class ConditionalPixelCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        ############################
        # TODO: Build conditional PixelCNN
        # - Use ConditionalMaskedConv2d instead of MaskedConv2d
        # - First layer: type 'A', later layers: type 'B'
        # - Add ReLU and BatchNorm as desired
        ############################

        self.layers = nn.ModuleList([
            ConditionalMaskedConv2d('A', 1, 64, kernel_size=7, padding=3, num_classes=num_classes),
            nn.ReLU(),

            ConditionalMaskedConv2d('B', 64, 64, kernel_size=7, padding=3, num_classes=num_classes),
            nn.ReLU(),

            ConditionalMaskedConv2d('B', 64, 64, kernel_size=7, padding=3, num_classes=num_classes),
            nn.ReLU(),

            ConditionalMaskedConv2d('B', 64, 64, kernel_size=7, padding=3, num_classes=num_classes),
            nn.ReLU(),

            ConditionalMaskedConv2d('B', 64, 64, kernel_size=7, padding=3, num_classes=num_classes),
            nn.ReLU(),
        ])

        ############################
        # TODO: Final output layer
        # - 1x1 conv mapping from hidden channels -> 1 channel
        ############################

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        

    def forward(self, x, y):
        """
        x: (N, 1, H, W) input images
        y: (N, num_classes) one-hot labels
        """
        ############################
        # TODO: Forward pass through layers
        # - For ConditionalMaskedConv2d, pass both (x, y)
        # - For other layers, just pass x
        ############################
        out = x
        for layer in self.layers:
            if isinstance(layer, ConditionalMaskedConv2d):
                out = layer(out, y)
            else:
                out = layer(out)

        out = self.output_conv(out)

        return out