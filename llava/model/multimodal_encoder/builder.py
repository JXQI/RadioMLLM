import os
from .clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    mm_vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(mm_vision_tower)
    if is_absolute_path_exists or mm_vision_tower.startswith("openai") or mm_vision_tower.startswith("laion"):
        return CLIPVisionTower(mm_vision_tower, args=vision_tower_cfg, **kwargs)

