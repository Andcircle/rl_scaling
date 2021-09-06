import os
import yaml
from datetime import datetime


def load_hypes(hypes_path):
    """
    Load hypes, setup training or testing folder, initialize logging
    :param hypes_path:
    :return:
    """
    with open(hypes_path, 'r') as rf:
        hypes = yaml.load(rf)
    run_time = datetime.now().strftime('%y_%m_%d_%H_%M')

    # means new training
    if 'output_dir' not in hypes:
        base_dir = os.path.dirname(os.path.realpath(hypes_path))
        log_dir = os.path.join(base_dir, '../logs')

        run_name = hypes['name'] + '_' + run_time
        output_dir = os.path.join(log_dir, run_name)
        hypes['output_dir'] = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hypes_save_path = output_dir + '/hypes'
        with open(hypes_save_path, 'w') as wf:
            yaml.dump(hypes, wf)

    return hypes



