import os
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader
from monodataset import MonoDataset
import PIL

class SimcolDataset(MonoDataset):
    def __init__(self, data_path, height=512, width=512, frame_idxs=[-3, 0, 3], num_scales=4, is_train=False):
        self.data_path = data_path
        self.full_res_shape = (475, 475)

        self.K = np.array([[227.60416,   0.0,       227.60416,   0.0],
                           [  0.0,       237.5,     237.5,       0.0],
                           [  0.0,       0.0,       1.0,         0.0],
                           [  0.0,       0.0,       0.0,         1.0]], dtype=np.float32)

        all_folders = []
        for root, dirs, files in os.walk(self.data_path):
            valid_frames = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]          
            if not valid_frames:
                continue
            
            relative_folder = os.path.relpath(root, self.data_path)
            
            # Đếm số lượng ảnh màu để biết chiều dài chuỗi video con này
            valid_frames.sort()
            all_folders.append((relative_folder, len(valid_frames)))

        self.filenames = []
        margin = max(abs(f) for f in frame_idxs if f != "s")

        for folder_path, total_frames in all_folders:
            full_path = os.path.join(self.data_path, folder_path) if not folder_path.startswith("/content") else folder_path
            if not os.path.exists(full_path):
                continue
                
            all_imgs = [f for f in os.listdir(full_path) if f.startswith("FrameBuffer_") and f.endswith(".png")]
            actual_total = len(all_imgs)

            if actual_total <= margin * 2:
                continue

            for img_idx in range(margin, actual_total - margin):
                self.filenames.append(f"{folder_path} {img_idx}")

        print(f"🎉 Quét đệ quy SimCOL thành công! Tìm thấy {len(all_folders)} chuỗi video con.")
        print(f"Tổng số mẫu đưa vào huấn luyện/đánh giá: {len(self.filenames)}")

        super(SimcolDataset, self).__init__(
            data_path=self.data_path,
            filenames=self.filenames,
            height=height,
            width=width,
            frame_idxs=frame_idxs,
            num_scales=num_scales,
            is_train=is_train
        )

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        if not folder.startswith("/content"):
            full_folder_path = os.path.join(self.data_path, folder)
        else:
            full_folder_path = folder

        f_str = f"{frame_index:04d}"
        image_path = os.path.join(full_folder_path, f"FrameBuffer_{f_str}.png")

        if not os.path.exists(image_path):
            image_path = os.path.join(full_folder_path, "FrameBuffer_0003.png")

        color = self.loader(image_path)
        if do_flip:
            color = color.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        if not folder.startswith("/content"):
            full_folder_path = os.path.join(self.data_path, folder)
        else:
            full_folder_path = folder

        f_str = f"{frame_index:04d}"
        depth_path = os.path.join(full_folder_path, f"Depth_{f_str}.png")

        if not os.path.exists(depth_path):
            depth_path = os.path.join(full_folder_path, "Depth_0003.png")

        depth_img = pil.open(depth_path)
        depth_img = depth_img.resize((self.width, self.height), PIL.Image.Resampling.NEAREST)        
        depth = np.array(depth_img, dtype=np.float32)

        if depth.max() > 255:
            depth = depth / 65535.0
        else:
            depth = depth / 255.0

        depth = depth * 0.20

        if do_flip:
            depth = np.fliplr(depth)
        return depth