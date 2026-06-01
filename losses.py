import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry import InverseWarp, transformation_from_parameters


# 1. SMOOTHNESS LOSS (L_s)
def edgeaware_smoothness_loss(disp, img):
    mean_disp = disp.mean(dim=(2, 3), keepdim=True) + 1e-7
    normalized_disp = disp / mean_disp

    grad_disp_x = torch.abs(normalized_disp[:, :, :, :-1] - normalized_disp[:, :, :, 1:])
    grad_disp_y = torch.abs(normalized_disp[:, :, :-1, :] - normalized_disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    loss_x = grad_disp_x * torch.exp(-grad_img_x)
    loss_y = grad_disp_y * torch.exp(-grad_img_y)

    return loss_x.mean() + loss_y.mean()


# 2. PHOTOMETRIC ERROR (pe)
class SSIM(nn.Module):
    """Khối bổ trợ tính toán SSIM bằng PyTorch thuần để phục vụ hàm pe"""
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, stride=1)
        self.mu_y_pool   = nn.AvgPool2d(3, stride=1)
        self.sig_x_pool  = nn.AvgPool2d(3, stride=1)
        self.sig_y_pool  = nn.AvgPool2d(3, stride=1)
        self.sig_xy_pool = nn.AvgPool2d(3, stride=1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class PhotometricLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(PhotometricLoss, self).__init__()
        self.alpha = alpha
        self.ssim = SSIM()

    def forward(self, pred_img, target_img):
        l1_loss = torch.abs(pred_img - target_img).mean(dim=1, keepdim=True)
        
        ssim_loss = self.ssim(pred_img, target_img).mean(dim=1, keepdim=True)
        
        pe_matrix = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
        return pe_matrix  


# 3. AUTO-MASKING
class AutoMasking(nn.Module):
    def __init__(self):
        super(AutoMasking, self).__init__()

    def forward(self, pe_warped_list, pe_identity_list):
        # min_t' pe(I_t, I_t'→t)
        all_pe_warped = torch.cat(pe_warped_list, dim=1)
        min_pe_warped, _ = torch.min(all_pe_warped, dim=1, keepdim=True)

        # min_t' pe(I_t, I_t')
        all_pe_identity = torch.cat(pe_identity_list, dim=1)
        all_pe_identity += torch.randn_like(all_pe_identity) * 1e-5
        min_pe_identity, _ = torch.min(all_pe_identity, dim=1, keepdim=True)

        mask = (min_pe_warped < min_pe_identity).float()

        return mask, min_pe_warped


# 4. MULTI-SCALE LOSS
class MultiScaleDepthLoss(nn.Module):
    def __init__(self, batch_size, height, width, alpha=0.85, smooth_weight=1e-3, frame_ids=[-3, 3]):
        super(MultiScaleDepthLoss, self).__init__()
        self.smooth_weight = smooth_weight
        self.frame_ids = frame_ids
        
        self.warp_module = InverseWarp(batch_size, height, width)
        self.photo_loss_module = PhotometricLoss(alpha=alpha)
        self.auto_mask_module = AutoMasking()

    def forward(self, inputs, outputs):
        total_loss = 0
        loss_metrics = {}

        for scale in [0, 1, 2, 3]:
            disp = outputs[("disp", scale)]
            depth = outputs[("depth", scale)]
            target_img = inputs[("color", 0, scale)]  # I_t
            
            smooth_loss = edgeaware_smoothness_loss(disp, target_img)
            
            pe_warped_list = []
            pe_identity_list = []
            
            for frame_id in self.frame_ids:
                source_img = inputs[("color", frame_id, 0)]  # Source image at orignal resolution I_t'
                
                axisangle = outputs[("axisangle", 0, frame_id)]      # [B, 1, 1, 3]
                translation = outputs[("translation", 0, frame_id)]  # [B, 1, 1, 3]
                axisangle_flatten = axisangle.reshape(-1, 3)
                translation_flatten = translation.reshape(-1, 3)
                T = transformation_from_parameters(
                    axisangle_flatten, translation_flatten, invert=(frame_id < 0)
                )      
                
                reconstructed_img = self.warp_module(
                    source_img, depth, inputs[("inv_K", scale)], inputs[("K", scale)], T
                )       
                         
                if scale > 0:
                    h_target, w_target = target_img.shape[2], target_img.shape[3]
                    reconstructed_img = F.interpolate(
                        reconstructed_img, size=(h_target, w_target), mode="bilinear", align_corners=False
                    )
                    source_img_scaled = F.interpolate(
                        source_img, size=(h_target, w_target), mode="bilinear", align_corners=False
                    )
                else:
                    source_img_scaled = source_img
                
                pe_warped_list.append(self.photo_loss_module(reconstructed_img, target_img))
                pe_identity_list.append(self.photo_loss_module(source_img_scaled, target_img))
            
            mask, min_pe_warped = self.auto_mask_module(pe_warped_list, pe_identity_list)
            
            final_photo_loss = (min_pe_warped * mask).mean()
            
            stage_loss = final_photo_loss + (self.smooth_weight * smooth_loss)
            total_loss += stage_loss
            
            loss_metrics[f"loss_stage/stage_{scale}"] = stage_loss.item()
            loss_metrics[f"photo_loss/stage_{scale}"] = final_photo_loss.item()
            loss_metrics[f"smooth_loss/stage_{scale}"] = smooth_loss.item()
            loss_metrics[f"mask_ratio/stage_{scale}"] = mask.mean().item() * 100

        final_loss = total_loss / 4
        return final_loss, loss_metrics