import os
from typing import Any, Dict, List

from core.constants import ROOT
from core.settings import settings
from utils.utils import read_npy_file


def load_imdbs_data() -> Dict[str, List[Dict[str, Any]]]:

    imdbs_path = settings.NPY_ROOT_PATH
    imdbs_full_path = os.path.join(ROOT, imdbs_path)
    imdbs_files = os.listdir(imdbs_full_path)
    data_dict = {}
    if not imdbs_files:
        raise FileNotFoundError(
            f"No files found in {imdbs_full_path}, please download them first."
        )
    for file in imdbs_files:
        if not file.endswith(".npy"):
            raise ValueError(f"File {file} is not a .npy file.")
        # load the npy file
        data = read_npy_file(os.path.join(imdbs_full_path, file))
        file_parts = file.split("_")
        split_part = file_parts[1].split(".")[0]
        data_dict[split_part] = data[1:]  # skip the first element, it is metadata
    return data_dict
