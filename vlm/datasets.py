import os
import json
from PIL import Image
from typing import Dict, List, Optional

import av
import numpy as np
from torch.utils.data import Dataset

def read_video_pyav(container, indices):
    """
    PyAV로 비디오 디코딩
    Args:
        container: av.container.input.InputContainer
        indices: 디코딩할 프레임 인덱스 리스트
    Returns:
        np.ndarray: (num_frames, height, width, 3)
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.user_key = user_key
        self.assistant_key = assistant_key

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")
            
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                images.append(
                    Image.open(image_path).convert("RGB")
                )
        
        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")
            
            for video_path in video_sources:
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)

                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                if total_frames == 0:
                    stream = container.streams.video[0]
                    total_frames = int(stream.duration * stream.average_rate)
                indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)

        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"

        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )