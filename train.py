import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from simcol_dataset import SimcolDataset
from losses import MultiScaleDepthLoss
from posenet import PoseCNN
from depthnet import DepthNetwork  

def train_pipeline():
    # 1. HYPERPARAMETERS & PATHS
    TRAIN_DATA_PATH = "/YOUR PATH HERE"
    VAL_DATA_PATH   = "YOUR PATH HERE"   
    
    BATCH_SIZE = 32
    HEIGHT = 512
    WIDTH = 512
    FRAME_IDS = [-3, 0, 3]               
    SOURCE_FRAME_IDS = [-3, 3]           
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SMOOTH_WEIGHT = 1e-3                 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System: {device}")

    # 2. DATALOADER 
    print("--- (Train) ---")
    train_dataset = SimcolDataset(
        TRAIN_DATA_PATH, height=HEIGHT, width=WIDTH, frame_idxs=FRAME_IDS, num_scales=4, is_train=True
    )
    
    print("\n--- (Val) ---")
    val_dataset = SimcolDataset(
        VAL_DATA_PATH, height=HEIGHT, width=WIDTH, frame_idxs=FRAME_IDS, num_scales=4, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    print(f"\n📊 Statistics: Train ({len(train_dataset)} frames) | Val ({len(val_dataset)} frames)")

    # 3. Network
    depth_net = DepthNetwork(min_depth=0.01, max_depth=0.2).to(device)
    pose_net = PoseCNN(num_input_frames=2).to(device)

    parameters_to_train = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = optim.Adam(parameters_to_train, lr=LEARNING_RATE)

    criterion = MultiScaleDepthLoss(
        batch_size=BATCH_SIZE, height=HEIGHT, width=WIDTH, smooth_weight=SMOOTH_WEIGHT, frame_ids=SOURCE_FRAME_IDS
    ).to(device)

    # 4. TRAINING LOOP
    for epoch in range(EPOCHS):
        
        depth_net.train()
        pose_net.train()
        epoch_train_loss = 0.0
        
        for batch_idx, inputs in enumerate(train_loader):
            for key in list(inputs.keys()):
                inputs[key] = inputs[key].to(device)

            optimizer.zero_grad()
            
            outputs = depth_net(inputs[("color_aug", 0, 0)])

            for frame_id in SOURCE_FRAME_IDS: 
                I_target = inputs[("color_aug", 0, 0)]
                I_source = inputs[("color_aug", frame_id, 0)]
                
                pose_inputs = torch.cat([I_target, I_source], dim=1)
                axisangle, translation = pose_net(pose_inputs)
                
                outputs[("axisangle", 0, frame_id)] = axisangle
                outputs[("translation", 0, frame_id)] = translation

            loss, metrics = criterion(inputs, outputs)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Mask Ratio S0: {metrics['mask_ratio/stage_0']:.1f}%")

        avg_train_loss = epoch_train_loss / len(train_loader)

        # 5. VALIDATION LOOP
        depth_net.eval()
        pose_net.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(val_loader):
                for key in list(inputs.keys()):
                    inputs[key] = inputs[key].to(device)

                outputs = depth_net(inputs[("color_aug", 0, 0)])

                for frame_id in SOURCE_FRAME_IDS:
                    pose_inputs = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", frame_id, 0)]], dim=1)
                    axisangle, translation = pose_net(pose_inputs)
                    
                    outputs[("axisangle", 0, frame_id)] = axisangle
                    outputs[("translation", 0, frame_id)] = translation

                val_loss, _ = criterion(inputs, outputs)
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"=======> END EPOCH {epoch} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} <=======\n")

        # 6. SAVE
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(depth_net.state_dict(), f"./checkpoints/depthnet_epoch_{epoch}.pth")
        torch.save(pose_net.state_dict(), f"./checkpoints/posenet_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_pipeline()