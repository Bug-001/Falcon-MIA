# output_manager.py

import os
import numpy as np
import json
import pickle
import functools
import hashlib
from typing import Any, Callable
from matplotlib import pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class FileManager:
    def __init__(self, name: str):
        self.base_dir = os.path.join('output', name)
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.image_dir = os.path.join(self.base_dir, 'image')
        self.cache_dir = os.path.join(self.base_dir, 'cache')
        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_json(self, filename: str, data: Any):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def save_image(self, filename: str, fig: plt.Figure):
        filepath = os.path.join(self.image_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

    def get_data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def get_image_path(self, filename: str) -> str:
        return os.path.join(self.image_dir, filename)
    
    def file_cache(self, func: Callable = None, *, enabled: bool = None):
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Determine if caching is enabled
                cache_enabled = enabled if enabled is not None else args[0].attack_config.get('cache_enabled', False)
                
                if not cache_enabled:
                    return f(*args, **kwargs)

                # Create a unique filename based on the function name and its arguments
                func_name = f.__name__
                args_str = str(args) + str(kwargs)
                filename = hashlib.md5((func_name + args_str).encode()).hexdigest() + '.pkl'
                cache_file = os.path.join(self.cache_dir, filename)

                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as file:
                        return pickle.load(file)
                else:
                    result = f(*args, **kwargs)
                    with open(cache_file, 'wb') as file:
                        pickle.dump(result, file)
                    return result
            return wrapper

        if func:
            return decorator(func)
        return decorator