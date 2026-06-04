import cv2
import torch


class AdhocVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, fps=1, shape=None, mean=None, std=None):
        self.video_path = video_path
        self.fps = fps

        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

        # Extract frame indices to sample based on fps
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.frame_indices = []
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(int(video_fps / fps), 1)

        # Calculate the starting frame at the middle of the second
        start_frame = int(video_fps / 2)
        self.frame_indices = list(range(start_frame, total_frames, frame_interval))

        cap.release()

    def __len__(self):
        return len(self.frame_indices)

    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(
                img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(
                f"Failed to read frame at index {frame_idx} from {self.video_path}"
            )

        img = self._preprocess(frame)
        frame_id = f"{self.video_path.replace('.mp4', '')}/frame_{frame_idx}.png"
        return frame_id, frame, img
