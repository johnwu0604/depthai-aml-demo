import json
from pathlib import Path

def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return str((dirname / relative_path).resolve())
    except FileNotFoundError:
        return None

def nn_json(file_path):
    NN_json = None
    with open(file_path) as f:
        NN_json = json.load(f)
        f.close()
    return NN_json

# blob_fpath            = relative_to_abs_path('../model/mobilenet-ssd.blob')
# blob_config_fpath     = relative_to_abs_path('../model/mobilenet-ssd.json')
blob_fpath            = relative_to_abs_path('../outputs/mobilenet-ssd-face-mask.blob')
blob_config_fpath     = relative_to_abs_path('../model/mobilenet-ssd-face-mask.json')


