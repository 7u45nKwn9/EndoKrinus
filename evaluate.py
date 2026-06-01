import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from simcol_dataset import SimcolDataset
from depthnet import DepthNetwork

def compute_errors(gt, pred):
    pred = torch.clamp(pred, min=1e-3)
    gt = torch.clamp(gt, min=1e-3)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean()).item()

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean()).item()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()
    sq_rel = torch.mean(((gt - pred) ** 2) / gt).item()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_pipeline(epoch_to_eval=49):
    TEST_DATA_PATH = "YOUR PATH HERE"  
    HEIGHT = 512
    WIDTH = 512
    FRAME_IDS = [-3, 0, 3]  
    BATCH_SIZE = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluate: {device}")

    test_dataset = SimcolDataset(
        TEST_DATA_PATH, height=HEIGHT, width=WIDTH, frame_idxs=FRAME_IDS, num_scales=4, is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"📊 Đang tiến hành đánh giá trên toàn bộ tập Test gồm: {len(test_dataset)} frames.")

    depth_net = DepthNetwork(min_depth=0.01, max_depth=0.2).to(device)
    weights_path = f"./checkpoints/depthnet_epoch_{epoch_to_eval}.pth"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Can't find weights: {weights_path}.")
        
    print(f"-> Downloading weights: {weights_path}")
    depth_net.load_state_dict(torch.load(weights_path, map_location=device))
    
    depth_net.eval()

    errors = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            input_color = inputs[("color", 0, 0)].to(device) 
            gt_depth = inputs["depth_gt"].to(device)             
            
            outputs = depth_net(input_color)
            pred_depth = outputs[("depth", 0)]                   
            
            for b in range(input_color.shape[0]):
                current_gt = gt_depth[b, 0]    
                current_pred = pred_depth[b, 0]  
                
                mask = current_gt > 0
                if mask.sum() == 0:
                    continue
                    
                gt_valid = current_gt[mask]
                pred_valid = current_pred[mask]
                
                err = compute_errors(gt_valid, pred_valid)
                errors.append(err)
                
    if len(errors) == 0:
        print("No frames")
        return

    duration = time.time() - start_time
    print(f"Time: {duration:.2f} seconds | FPS: {len(test_dataset)/duration:.1f} frames/second")

    mean_errors = np.array(errors).mean(0)
    print("\n" + "="*70)
    print(f" Evaluate DEPTHNET (EPOCH {epoch_to_eval}) on test set")
    print("="*70)
    
    metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1 (1.25)", "a2 (1.25²)", "a3 (1.25³)"]
    print(f"{'Metric':<18} | {'Mean':<25} | {'Status':<20}")
    print("-"*70)
    print(f"{metric_names[0]:<18} | {mean_errors[0]:<25.4f} | Lower is better ↓")
    print(f"{metric_names[1]:<18} | {mean_errors[1]:<25.4f} | Lower is better ↓")
    print(f"{metric_names[2]:<18} | {mean_errors[2]:<25.4f} m | Lower is better ↓") 
    print(f"{metric_names[3]:<18} | {mean_errors[3]:<25.4f} | Lower is better ↓")
    print(f"{metric_names[4]:<18} | {mean_errors[4]*100:<23.2f}% | Higher is better ↑")
    print(f"{metric_names[5]:<18} | {mean_errors[5]*100:<23.2f}% | Higher is better ↑")
    print(f"{metric_names[6]:<18} | {mean_errors[6]*100:<23.2f}% | Higher is better ↑")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate_pipeline(epoch_to_eval=49)