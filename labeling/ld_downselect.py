import numpy as np
import argparse

def parse_args():
    """
    Parse command line arguments for downselecting landmarks from Sareana.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Labeling Landsat Data')
    parser.add_argument('--path', type=str, required=True, help='Path to all landmarks file')
    parser.add_argument('--num_landmarks', type=int, required=True, help='Number of landmarks to downselect to')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save downselected landmarks')
    parser.add_argument('--scales', type=float, default=None, nargs='+', help='Scales to downselect from')
    parsed_args = parser.parse_args()
    if parsed_args.scales is None:
        parsed_args.scales = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    return parsed_args

print('Downselecting landmarks...')


args = parse_args()
all_landmarks = np.load(args.path)
scales = args.scales
num_classes = args.num_landmarks
tot_landmarks = 0
for scale in scales:
    tot_landmarks += len(all_landmarks[all_landmarks[:,-2] == scale])
thresh = (1 - num_classes/tot_landmarks) * 100

lds = []
for scale in scales:
    lds_at_scale = all_landmarks[all_landmarks[:,-2] == scale]
    threshold_val = np.percentile(lds_at_scale[:,-1], thresh)
    lds_at_scale = lds_at_scale[lds_at_scale[:,-1] > threshold_val]
    lds.append(lds_at_scale)
lds_ds = np.vstack(lds)
lds_corners = lds_ds
np.save(args.output_path, lds_corners)
print('Downselected landmarks saved to', args.output_path)
print('Downselected to', len(lds_ds), 'landmarks from', tot_landmarks, 'landmarks')