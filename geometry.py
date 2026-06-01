import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def rot_from_axisangle(vec):
    angle = torch.norm(vec, p=2, dim=1, keepdim=True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[:, 0:1]
    y = axis[:, 1:2]
    z = axis[:, 2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 3, 3), device=vec.device)

    rot[:, 0, 0] = (x * xC + ca).squeeze(-1)
    rot[:, 0, 1] = (xyC - zs).squeeze(-1)
    rot[:, 0, 2] = (zxC + ys).squeeze(-1)
    rot[:, 1, 0] = (xyC + zs).squeeze(-1)
    rot[:, 1, 1] = (y * yC + ca).squeeze(-1)
    rot[:, 1, 2] = (yzC - xs).squeeze(-1)
    rot[:, 2, 0] = (zxC - ys).squeeze(-1)
    rot[:, 2, 1] = (yzC + xs).squeeze(-1)
    rot[:, 2, 2] = (z * zC + ca).squeeze(-1)

    return rot


def get_translation_matrix(translation_vector):
    T = torch.zeros((translation_vector.shape[0], 4, 4), device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1.0
    T[:, 1, 1] = 1.0
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0
    T[:, :3, 3, None] = t

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle) # R: [B, 3, 3]
    t = translation.clone()           # t: [B, 3]

    if invert:
        R = R.transpose(1, 2) 
        t = -torch.matmul(R, t.unsqueeze(-1)).squeeze(-1)

    M = torch.zeros((axisangle.shape[0], 4, 4), device=axisangle.device)
    M[:, :3, :3] = R
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    
    return M


class BackprojectDepth(nn.Module):
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        j_indices = torch.arange(self.height, dtype=torch.float32)
        i_indices = torch.arange(self.width, dtype=torch.float32)
        j, i = torch.meshgrid(j_indices, i_indices, indexing='ij')

        pix_coords = torch.stack([i.reshape(-1), j.reshape(-1), torch.ones_like(i.reshape(-1))], dim=0)
        pix_coords = pix_coords.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.pix_coords = nn.Parameter(pix_coords, requires_grad=False)

        ones = torch.ones(self.batch_size, 1, self.height * self.width, dtype=torch.float32)
        self.ones = nn.Parameter(ones, requires_grad=False)

    def forward(self, depth, inv_K):
        B, _, H, W = depth.shape 
        
        if not hasattr(self, 'dynamic_grid') or self.dynamic_grid.shape[-1] != H * W or self.dynamic_grid.device != depth.device:
            meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = torch.from_numpy(id_coords).to(depth.device)
            
            ones = torch.ones(1, 1, H, W).to(depth.device)
            pix_coords = torch.cat([id_coords.unsqueeze(0), ones], 1)
            pix_coords = pix_coords.view(1, 3, -1)
            pix_coords = pix_coords.repeat(B, 1, 1)
            self.dynamic_grid = pix_coords
            
        cam_points = torch.matmul(inv_K[:, :3, :3], self.dynamic_grid)
        cam_points = depth.view(B, 1, -1) * cam_points
        
        ones_pad = torch.ones(B, 1, H * W).to(depth.device)

        return torch.cat([cam_points, ones_pad], 1)


class Project3D(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        B = points.shape[0]
        
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
        
        num_points = pix_coords.shape[-1] 
        current_size = int(np.sqrt(num_points)) 
        
        pix_coords = pix_coords.view(B, 2, current_size, current_size)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        
        return pix_coords


class InverseWarp(nn.Module):
    def __init__(self, batch_size, height, width):
        super(InverseWarp, self).__init__()

        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.project_3d = Project3D(batch_size, height, width)

    def forward(self, source_img, depth, inv_K, K, T):
        world_points = self.backproject_depth(depth, inv_K)
        
        pix_coords = self.project_3d(world_points, K, T)
        
        reconstructed_img = F.grid_sample(source_img, pix_coords, padding_mode="border", align_corners=True)
        
        return reconstructed_img