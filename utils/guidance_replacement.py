import os, sys
import logging
import numpy as np
from PIL import Image
import gradio as gr
import torch
import random
import torch.nn.functional as F
from einops import rearrange
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def replace_attn_with_move_object_against_single_point(attn_scores, region_mask, threshold=0.5, sharpness=1, tokens=[0], f_repl=10, f_margin=0.5, clamp=1, region_exclusion=1, theta=0.15, is_debug=False):
    """Move object attention away from a region using repulsive and margin forces.
    
    Args:
        attn_scores: Attention scores tensor (B, HW, cross_attention_token) or (B, HW, HW) for self attention
        region_mask: Region mask tensor
        threshold: Threshold for soft thresholding (0-1)
        sharpness: Sharpness parameter for thresholding
        tokens: List of token indices to modify
        f_repl: Repulsive force coefficient
        f_margin: Margin force coefficient 
        clamp: Maximum force magnitude
        region_exclusion: Region exclusion strength (0-1)
        theta: Conflict detection threshold
        
    Returns:
        Modified attention scores tensor
    """
    # Handle different input shapes
    if len(attn_scores.shape) == 3:
        B, HW, C = attn_scores.shape
        heads = 1
        attn_scores = attn_scores.unsqueeze(1)  # Add head dimension
    else:
        B, heads, HW, C = attn_scores.shape
        
    # Calculate H,W from HW
    downscale_h = round(HW ** 0.5)
    H, W = downscale_h, HW // downscale_h
    
    # 检查 tokens 是否为 range 类型，如果是且超出了最后一个维度，则直接截断
    if isinstance(tokens, range) and tokens.stop > C:
        tokens = range(tokens.start, C)
        if is_debug:
            print(f"Tokens range truncated to {tokens}")
    
    K = len(tokens)
    # Extract attention maps for target tokens
    if is_debug:
        print(f"attn_scores shape: {attn_scores.shape}, num_tokens: {K}, tokens: {tokens}")
    attn_map = attn_scores[..., tokens].detach().clone()
    
    # Mean over heads if present
    if heads > 1:
        attn_map = attn_map.view(B, heads, H, W, K).mean(dim=1)
    else:
        attn_map = attn_map.view(B, H, W, K)

    # Threshold attention maps
    attn_map = soft_threshold(attn_map, threshold=threshold, sharpness=sharpness)   # Shape: (B, H, W, K)

    # Resize region mask if needed
    if region_mask.shape[1:3] != attn_map.shape[1:3]:
        region_mask = region_mask.permute(0, 3, 1, 2)
        region_mask = F.interpolate(region_mask, size=(attn_map.shape[1:3]), mode='bilinear') 
        region_mask = region_mask.permute(0, 2, 3, 1)   # Shape: (B, H, W, 1)

    # Rest of the implementation remains same...
    region_mask_centroid = calculate_centroid(region_mask)
    conflicts = detect_conflict(attn_map, region_mask, theta)
    
    if not torch.any(conflicts > 0.01):
        return attn_scores
        
    # Continue with existing logic...
    centroids = calculate_centroid(attn_map)
    displ_force = displacement_force(attn_map, centroids, region_mask_centroid, f_repl, f_margin, clamp)
    displ_force = displ_force * conflicts.unsqueeze(-1)
    if is_debug:
        print(f"displ_force shape: {displ_force.shape}, num_tokens: {K}, {displ_force}")
        print(f"conflicts shape: {conflicts.shape}, num_tokens: {K}, {conflicts}")
    # Expand for heads
    displ_force = torch.stack([displ_force]*heads, dim=1).view(B*heads, K, -1)
    centroids = torch.stack([centroids]*heads, dim=1).view(B*heads, K, -1)
    
    # Apply modifications
    output_attn_map = attn_scores[..., tokens].detach().clone()
    modified_attn_map, _ = apply_displacements(output_attn_map, centroids, displ_force , H , W)
    modified_attn_map=modified_attn_map.view(B,heads,H*W,-1)
    # Region exclusion
    region_mask = region_mask.view(B, H*W, 1)
    region_mask = region_mask.unsqueeze(1).expand(-1, heads, -1, K)
    modified_attn_map = region_exclusion * modified_attn_map + (1-region_exclusion) * modified_attn_map * (1-region_mask)
    
    # Update output
    tcg_attnmap = attn_scores.detach().clone()
    tcg_attnmap[..., tokens] = modified_attn_map
    tcg_attnmap = tcg_attnmap.view(B, heads, HW, C)
    
    # Remove head dimension if input was 3D
    if len(attn_scores.shape) == 3:
        tcg_attnmap = tcg_attnmap.squeeze(1)
        
    return tcg_attnmap





def displacement_force(attention_map, verts, target_pos, f_rep_strength, f_margin_strength, clamp=0):
    """ Given a set of vertices, calculate the displacement force given by the sum of margin force and repulsive force.
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (B, C, 2)
        target : torch.Tensor - The vertices of the targets. Shape: (2)
        f_rep_strength: float - The strength of the repulsive force
        f_margin_strength: float - The strength of the margin force
    Returns:
        torch.Tensor - The displacement force for each vertex. Shape: (B, C, 2)
    """
    B, H, W, C = attention_map.shape
    f_clamp = lambda x: x
    if clamp > 0:
        clamp_min = -clamp
        clamp_max = clamp
        f_clamp = lambda x: torch.clamp(x, min=clamp_min, max=clamp_max)

    f_rep = f_clamp(repulsive_force(f_rep_strength, verts, target_pos))
    f_margin = f_clamp(margin_force(f_margin_strength, H, W, verts))

    #logger.debug(f"Repulsive force: {debug_coord(f_rep)}, Margin force: {debug_coord(f_margin)}")
    return f_rep + f_margin


def normalize_map(attnmap):
    """ Normalize the attention map over the channel dimension
    Arguments:
        attnmap: torch.Tensor - The attention map to normalize. Shape: (B, HW, C)
    Returns:
        torch.Tensor - The attention map normalized to (0, 1). Shape: (B, HW, C)
    """
    flattened_attnmap = attnmap.transpose(-1, -2)
    min_val = torch.min(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
    max_val = torch.max(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
    normalized_attn = (flattened_attnmap - min_val) / ((max_val - min_val) + torch.finfo(attnmap.dtype).eps)
    normalized_attn = normalized_attn.transpose(-1, -2)
    return normalized_attn


def soft_threshold(attention_map, threshold=0.5, sharpness=10):
    """ Soft threshold the attention map channels based on the given threshold. Derived from arXiv:2306.00986
    Arguments:
        attention_map: torch.Tensor - The attention map to threshold. Shape: (B, H, W, C)
        threshold: float - The threshold value between 0.0 and 1.0 relative to the minimum/maximum attention value
        sharpness: float - The sharpness of the thresholding function
    Returns:
        torch.Tensor - The attention map thresholded over all C. Shape: (B, H, W, C)
    """
    def _normalize_map(attnmap):
        """ Normalize the attention map over the channel dimension
        Arguments:
            attnmap: torch.Tensor - The attention map to normalize. Shape: (B, H, W, C)
        Returns:
            torch.Tensor - The attention map normalized to (0, 1). Shape: (B, H, W, C)
        """
        B, H, W, C = attnmap.shape
        flattened_attnmap = attnmap.view(attnmap.shape[0], H*W, attnmap.shape[-1]) # B, H*W, C
        min_val = torch.min(flattened_attnmap, dim=1).values.unsqueeze(1) # (B, 1, C)
        max_val = torch.max(flattened_attnmap, dim=1).values.unsqueeze(1) # (B, 1, C)
        normalized_attn = (flattened_attnmap - min_val) / ((max_val - min_val) + torch.finfo(attnmap.dtype).eps)
        normalized_attn = normalized_attn.view(B, H, W, C) # B, H*W, C
        #normalized_attn = normalized_attn.view(B, H, W, C)
        return normalized_attn
    threshold = max(0.0, min(1.0, threshold))
    normalized_attn = _normalize_map(attention_map)
    normalized_attn = _normalize_map(torch.sigmoid(sharpness * (normalized_attn - threshold)))
    return normalized_attn


def distances_to_nearest_edges(verts, h, w):
    """ Calculate the distances and direction to the nearest edge bounded by (H, W) for each channel's vertices 
    Arguments:
        verts: torch.Tensor - The vertices. Shape: (B, C, 2), where the last 2 dims are (y, x)
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor, torch.Tensor:
          - The minimum distance of each vertex to the nearest edge. Shape: (B, C, 1)
          - The direction to the nearest edge. Shape: (B, C, 4, 2), where the last 2 dims are (y, x)
    """
    # y axis is 0!
    y = verts[..., 0] # (B, C, 2)
    x = verts[..., 1] # (B, C, 2)
    B, C, _ = verts.shape
    # distances = torch.stack([y, y-h, x , x-w], dim=-1)
    distances = torch.stack([y, h - y, x, w - x], dim=-1) # (B, C, 4)
    
    directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]]).view(1, 1, 4, 2).repeat(B, C, 1, 1) # (4, 2) -> (B, C, 4, 2)
    directions = directions.to(verts.device)
    
    return distances, directions


def min_distance_to_nearest_edge(verts, h, w):
    """ Calculate the distances and direction to the nearest edge bounded by (H, W) for each channel's vertices 
    Arguments:
        verts: torch.Tensor - The vertices. Shape: (B, C, 2), where the last 2 dims are (y, x)
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor, torch.Tensor:
          - The minimum distance of each vertex to the nearest edge. Shape: (B, C)
          - The direction to the nearest edge. Shape: (B, C, 2), where the last 2 dims are (y, x)
    """
    y = verts[..., 0] # y-axis is 0!
    x = verts[..., 1]
    
    # Calculate distances to the edges (y, h-y, x, w-x)
    # y: distance to top edge
    # h - y: distance to bottom edge
    # x: distance to left edge
    # w - x: distance to right edge
    distances = torch.abs(torch.stack([y, h - y, x, w - x], dim=-1))
    
    # Find the minimum distance and the corresponding closest edge
    min_distances, min_indices = distances.min(dim=-1)
    
    # Map edge indices to direction vectors
    directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]]).to(verts.device)
    nearest_edge_dir = directions[min_indices]
    
    return min_distances, nearest_edge_dir


def margin_force(strength, H, W, verts):
    """ Margin force calculation
    Arguments:
        strength: float - The margin force coefficient
        H: float - The height of the image
        W: float - The width of the image
        verts: torch.Tensor - The vertices of the attention map. Shape: (B, C, 2)
    Returns:
        torch.Tensor - The force for each vertex. Shape: (B, C, 2)
    """
    distances, edge_dirs = distances_to_nearest_edges(verts, H, W) # (B, C, 4), (B, C, 4, 2)
    distances = distances.unsqueeze(-1)

    #distances = distances.unsqueeze(-1) # (B, C, 1)
    force_multiplier = -strength / (distances ** 2 + torch.finfo(distances.dtype).eps)
    forces = force_multiplier * edge_dirs # (B, C, 4, 2)
    forces = forces.sum(dim=-2) # (B, C, 2) # sum over the 4 directions to get total force

    return forces

def warping_force(attention_map, verts, displacements, h, w):
    """ Rescales the attention map based on the displacements. Expects a batch size of 1 to operate on all channels at once.
    Arguments:
        attention_map: torch.Tensor - The attention map to update. Shape: (1, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (1, C, 2)
        displacements: torch.Tensor - The displacements to apply. Shape: (1, C, 2), where the last 2 dims are the translation by [Y, X]
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor - The updated attention map. Shape: (B, H, W, C)
    """
    B, H, W, C = attention_map.shape

    old_centroids = verts # (B, C, 2)
    new_centroids = old_centroids + displacements * torch.tensor([h, w], device=attention_map.device).view(1, 1, 2) # (B, C, 2)

    # check if new_centroids are out of bounds
    min_bounds = torch.tensor([0, 0], dtype=torch.float32, device=attention_map.device)
    max_bounds = torch.tensor([h-1, w-1], dtype=torch.float32, device=attention_map.device)
    oob_new_centroids = torch.clamp(new_centroids, min_bounds, max_bounds)

    # diferenct between old and new centroids
    correction = oob_new_centroids - new_centroids
    new_centroids = new_centroids + correction

    s_y = (h - 1)/new_centroids[..., 0] # (B, C) 
    s_x = (w - 1)/new_centroids[..., 1] # (B, C) 
    torch.clamp_max(s_y, 1.0, out=s_y)
    torch.clamp_max(s_x, 1.0, out=s_x)
    if torch.any(s_x < 0.99) or torch.any(s_y < 0.99):
        #logger.debug(f"Scaling factor: {s_x}, {s_y}")
        pass

    # displacements

    #correction+new=oob
    #dis-new=-old
    #oob-old=dis+correction
    o_new = displacements * torch.tensor([h, w], device=attention_map.device).view(1, 1, 2) + correction

    # construct affine transformation matrices (sx, 0, delta_x - o_new_x), (0, sy, delta_y - o_new_y)
    theta = torch.zeros((C, 2, 3), dtype=torch.float32, device=attention_map.device)
    # 设置缩放因子
    theta[:, 0, 0] = s_x  # 设置 s_x 到 theta 的第一行第一列
    theta[:, 1, 1] = s_y  # 设置 s_y 到 theta 的第二行第二列
    # 设置平移量
    theta[:, 0, 2] = o_new[..., 1] / w  # 设置 X 方向平移量
    theta[:, 1, 2] = o_new[..., 0] / h  # 设置 Y 方向平移量
    # 生成仿射变换网格
    grid = F.affine_grid(theta, [C, B, H, W], align_corners=False)

    attention_map = attention_map.permute(3, 0, 1, 2) # (B, H, W, C) -> (C,B,H, W)
    attention_map = attention_map.to(torch.float32)
    out_attn_map = F.grid_sample(attention_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    out_attn_map = out_attn_map.permute(1, 2, 3, 0) # (C, B, H, W) -> (B, H, W, C)
    out_attn_map=out_attn_map.view(B,H*W,C).squeeze(0)
    # rescale centroids to pixel space
    return out_attn_map, new_centroids



def repulsive_force(strength, pos_vertex, pos_target):
    """ Repulsive force repels the vertices in the direction away from the target 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        strength: float - The global force coefficient
        pos_vertex: torch.Tensor - The position of the vertex. Shape: (B, C, 2)
        pos_target: torch.Tensor - The position of the target. Shape: (2)
    Returns:
        torch.Tensor - The force away from the target. Shape: (B, C, 2)
    """
    d_pos = pos_vertex - pos_target # (B, C, 2)
    d_pos_norm = d_pos.norm() + torch.finfo(d_pos.dtype).eps # normalize the direction
    #d_pos_norm = d_pos.norm(dim=-1, keepdim=True) + torch.finfo(d_pos.dtype).eps # normalize the direction
    d_pos = d_pos / d_pos_norm
    # d_pos /= d_pos_norm
    force = -(strength ** 2)
    return force*d_pos_norm / d_pos


def multi_target_force(attention_map, omega, xi, pos_vertex, pos_target):
    """ Multi-target force calculation
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        omega: torch.tensor - Coefficients for balancing forces amongst j targets
        xi: float - The global force coefficient
        pos_vertex: torch.Tensor - The position of the vertex. Shape: (B, C, 2)
        pos_target: torch.Tensor - The position of the target. Shape: (B, C, 2)
    Returns:
        torch.Tensor - The multi-target force. Shape: (B, C, 2)
    """
    force = -xi ** 2
    pass


def calculate_region(attention_map):
    """ Given an attention map of shape [B, H, W, C], calculate a bounding box over each C
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the bounding box. Shape: (B, H, W, C)
    Returns:
        torch.Tensor - The bounding box of the region. Shape: (B, C, 4), where the last 4 dims are (y, x, a, b)
        y, x: The top left corner of the bounding box
        a, b: The height and width of the bounding box
    """
    B, H, W, C = attention_map.shape
    # Calculate the sum of attention map along the height and width dimensions
    sum_map = attention_map.sum(dim=(1, 2)) # (B, C)
    # Find the indices of the maximum attention value for each channel
    max_indices = sum_map.argmax(dim=1, keepdim=True) # (B, C)
    # Initialize the bounding box tensor
    bounding_box = torch.zeros((B, C, 4), dtype=torch.int32, device=attention_map.device)
    # Iterate over each channel
    for batch_idx in range(B):
        for channel_idx in range(C):
            # Calculate the row and column indices of the maximum attention value
            row_index = max_indices[batch_idx, channel_idx] // W
            col_index = max_indices[batch_idx, channel_idx] % W
            # Calculate the top left corner coordinates of the bounding box
            y = max(0, row_index - 1)
            x = max(0, col_index - 1)
            # Calculate the height and width of the bounding box
            a = min(H - y, row_index + 2) - y
            b = min(W - x, col_index + 2) - x
            # Store the bounding box coordinates in the tensor
            bounding_box[batch_idx, channel_idx] = torch.tensor([y, x, a, b])
    return bounding_box


def calculate_centroid(attention_map):
    """ Calculate the centroid of the attention map 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the centroid. Shape: (B, H, W, C)
    Returns:
        torch.Tensor - The centroid of the attention map. Shape: (B, C, 2), where the last 2 dims are (y, x)
    """
    # necessary to avoid inf
    attention_map = attention_map.to(torch.float32)
    B, H, W, C = attention_map.shape

    # Create tensors of the y and x coordinates
    y_coords = torch.arange(H, dtype=attention_map.dtype, device=attention_map.device).view(1, H, 1, 1)
    x_coords = torch.arange(W, dtype=attention_map.dtype, device=attention_map.device).view(1, 1, W, 1)

    # Calculate the weighted sums of the coordinates
    weighted_sum_y = torch.sum(y_coords * attention_map, dim=[1, 2])
    weighted_sum_x = torch.sum(x_coords * attention_map, dim=[1, 2])

    # Calculate the total weights
    total_weights = torch.sum(attention_map, dim=[1, 2]) + torch.finfo(attention_map.dtype).eps

    # Calculate the centroids
    centroid_y = weighted_sum_y / total_weights
    centroid_x = weighted_sum_x / total_weights

    # Combine x and y centroids
    centroids = torch.stack([centroid_y, centroid_x], dim=-1) 
    
    return centroids


def detect_conflict(attention_map, region, theta):
    """
    Detect conflict in an attention map with respect to a designated region in PyTorch.
    Parameters:
    attention_map (torch.Tensor): Attention map of shape (B, H, W, K).
    region (torch.Tensor): Binary mask of shape (B, H, W, 1) indicating the region of interest.
    theta (float): Threshold value.
    Returns:
    torch.Tensor: Conflict detection result of shape (B, K), with values 0 or 1 indicating conflict between tokens and the region.
    """
    # Ensure region is the same shape as the spatial dimensions of attention_map
    assert region.shape[1:3] == attention_map.shape[1:3], "Region mask must match spatial dimensions of attention map"
    # Calculate the mean attention within the region
    #region = region.unsqueeze(-1) # Add channel dimension: (B, H, W) -> (B, H, W, 1)
    # HACK: fixme
    if region.dim() != attention_map.dim():
        attention_in_region = attention_map * region.unsqueeze(-1) # Element-wise multiplication
    else:
        attention_in_region = attention_map * region
    #mean_attention_in_region = attention_in_region[attention_in_region > 0] 
    mean_attention_in_region = torch.sum(attention_in_region, dim=(1, 2)) / torch.sum(region, dim=(1, 2)) # Mean over (H, W)
    # Compare with threshold theta
    conflict = (mean_attention_in_region > theta).float() # Convert boolean to float (0 or 1)
    return conflict

def apply_displacements(attention_map, verts, displacements ,H , W):
    """ Update the attention map based on the displacements.
    The attention map is updated by displacing the attention values based on the displacements. 
    - Areas that are displaced out of the attention map are discarded.
    - Areas that are displaced into the attention map are initialized with zeros.
    Arguments:
        attention_map: torch.Tensor - The attention map to update. Shape: (B, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (B, C, 2)
        displacements: torch.Tensor - The displacements to apply in pixel space. Shape: (B, C, 2), where the last 2 dims are the translation by [Y, X]
    Returns:
        torch.Tensor - The updated attention map. Shape: (B, H, W, C)
    """
    B,heads,_,C = attention_map.shape
    reshaped_tensor = attention_map.view(B, -1, H, W, C)
    # 然后合并 B 和 HEADS 维度
    final_tensor = reshaped_tensor.view(-1, H, W, C)
    out_attn_map = attention_map.detach().clone()
    out_verts = verts.detach().clone()
    # apply displacements
    for batch_idx in range(B):
        out_attn_map[batch_idx], out_verts[batch_idx] = warping_force(final_tensor[batch_idx].unsqueeze(0), verts[batch_idx].unsqueeze(0), displacements[batch_idx], H, W)
        out_attn_map[batch_idx] = out_attn_map[batch_idx].squeeze(0)

    return out_attn_map, out_verts



