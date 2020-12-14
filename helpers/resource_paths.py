from pathlib import Path

def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return str((dirname / relative_path).resolve())
    except FileNotFoundError:
        return None

blob_fpath            = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.blob.sh14cmx14NCE1')
blob_config_fpath     = relative_to_abs_path('../resources/nn/mobilenet-ssd/mobilenet-ssd.json')
