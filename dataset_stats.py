from track_utils import *
import os

DATA_PATH = 'dancetrack_stats/dancetrack'
SPLITS = ['train', 'val']
# 'test' does not contain ground truth

def get_track_names(seq_map_path):
    track_names = []
    with open(seq_map_path) as f:
        for line in f:
            if line.strip() != "name":
                track_names.append(line.strip())
    return track_names

if __name__ == '__main__':
    # dataset stat variables
    num_sequences = 0
    total_frames = 0
    num_frames_in_sequence = []
    num_instances = 0
    total_pairs_to_annotate = 0
    num_pairs_to_annotate_in_sequence = []
    total_overlapping_pairs = 0
    num_overlapping_pairs_in_sequence = []

    for split in SPLITS:
        split_dir = os.path.join(DATA_PATH, split)
        seq_map_path = os.path.join(DATA_PATH, split + '_seqmap.txt')
        tracks = get_track_names(seq_map_path)
        for track in tracks:
            seq_dir = os.path.join(split_dir, track)
            data_sequence = Sequence(seq_dir)




    """
    # TODO
    1. Dataloader for train test val data
    2. Running totals for
        a. # of sequences
        b. # of frames
        c. # of instances 
        d. # of pairs (needed to annotate for heuristic)
        e. # of overlapping pairs
            hint: use gt to calculate
    """
    
