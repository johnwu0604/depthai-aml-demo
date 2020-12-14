from . import resource_paths

config = {
    'streams': ['metaout', 'previewout'],
    'ai': {
        'blob_file': resource_paths.blob_fpath,
        'blob_file_config': resource_paths.blob_config_fpath,
        'camera_input': 'rgb',
        'shaves': 14,
        'cmx_slices': 14,
        'NN_engines': 1,
        'calc_dist_to_bb': True,
        'keep_aspect_ratio': True
    },
    'camera':{
        'rgb': {
            'resolution_h': 1080, # possible - 1080, 2160, 3040
            'fps': 30,
        },
        'mono':{
            'resolution_h': 400, # possible - 400, 720, 800
            'fps': 30,
        }
    }
}