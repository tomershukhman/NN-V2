    def generate_anchors(self, feature_map_size, device):
        """
        Generate anchor boxes for the given feature map size.
        Maps feature map coordinates to the original image space.
        """
        fm_height, fm_width = feature_map_size
        
        # Calculate stride based on input image size and feature map size
        stride_x = self.input_size[0] / fm_width
        stride_y = self.input_size[1] / fm_height
        
        # Generate grid centers in image coordinates
        shifts_x = torch.arange(0, fm_width, device=device, dtype=torch.float32) * stride_x + stride_x / 2
        shifts_y = torch.arange(0, fm_height, device=device, dtype=torch.float32) * stride_y + stride_y / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Generate base anchors around (0, 0)
        base_anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                h = scale
                w = scale * ratio
                base_anchors.append([-w/2, -h/2, w/2, h/2])
                
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        
        # For each grid cell center, add all base anchors
        all_anchors = []
        for cy, cx in zip(shift_y.flatten(), shift_x.flatten()):
            # Move base anchors to grid cell center
            cell_anchors = base_anchors.clone()
            cell_anchors[:, [0, 2]] += cx  # Add cx to x coordinates
            cell_anchors[:, [1, 3]] += cy  # Add cy to y coordinates
            all_anchors.append(cell_anchors)
        
        # Stack all anchors into single tensor
        all_anchors = torch.cat(all_anchors, dim=0)
        
        # Clip to image bounds
        all_anchors = torch.stack([
            all_anchors[:, 0].clamp(min=0, max=self.input_size[0]),
            all_anchors[:, 1].clamp(min=0, max=self.input_size[1]),
            all_anchors[:, 2].clamp(min=0, max=self.input_size[0]),
            all_anchors[:, 3].clamp(min=0, max=self.input_size[1])
        ], dim=1)
        
        # Check for invalid boxes
        invalid = (all_anchors[:, 2] <= all_anchors[:, 0]) | (all_anchors[:, 3] <= all_anchors[:, 1])
        if invalid.any():
            # Remove invalid anchors
            valid_mask = ~invalid
            all_anchors = all_anchors[valid_mask]

        return all_anchors