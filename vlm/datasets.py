import os
import json
from PIL import Image
from typing import Dict, List, Optional

from torch.utils.data import Dataset

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: Optional[str] = None,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
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
                raise ValueError(f"Invalid image source type: {type(source["image"])}")
            
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                images.append(
                    Image.open(image_path).convert("RGB")
                )
        
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
            conversations=convs,
            system_prompt=system_prompt
        )