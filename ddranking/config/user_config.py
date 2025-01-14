import yaml
import json
from typing import Dict, Any
from torchvision import transforms

class Config:
    """Configuration object to manage individual configurations."""
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with a configuration dictionary."""
        self.config = config or {}

    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from a YAML or JSON file."""
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use YAML or JSON.")
        return cls(config)
    
    def load_transforms_from_yaml(self, values):
        if values is None:
            return None
        transform_list = [] 
        for transform in values:
            name = transform["name"]
            args = transform.get("args", [])
            if isinstance(args, dict):
                transform_list.append(getattr(transforms, name)(**args))
            else:
                transform_list.append(getattr(transforms, name)(*args))
        
        return transforms.Compose(transform_list)

    def get(self, key: str, default: Any = None):
        """Get a value from the config."""
        if key == "custom_train_trans":
            return self.load_transforms_from_yaml(self.config["custom_train_trans"])
        elif key == "custom_val_trans":
            return self.load_transforms_from_yaml(self.config["custom_val_trans"])
        elif key == "im_size":
            return tuple(self.config.get("im_size", default))
        
        return self.config.get(key, default)

    def update(self, overrides: Dict[str, Any]):
        """Update the configuration with overrides."""
        self.config.update(overrides)

    def __repr__(self):
        return f"Config({self.config})"