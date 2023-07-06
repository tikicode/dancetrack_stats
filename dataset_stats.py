from track_utils import *
import os
import json

DATA_PATH = 'dancetrack_stats/dancetrack'
SPLITS = ['train', 'val', 'test'] # 'test' does not contain ground truth
THRESH = 0.8  # threshold for overlap

def get_track_names(seq_map_path):
    track_names = []
    with open(seq_map_path) as f:
        for line in f:
            if line.strip() != "name":
                track_names.append(line.strip())
    track_nums = [int(name[10:]) for name in track_names]
    return track_names, track_nums

def get_number_of_frames(img_path):
    num_frames = 0
    for file in os.listdir(img_path):
        if file.endswith('.jpg'):
            num_frames += 1
    return num_frames

def get_num_pairs_to_annotate_in_sequence(data_sequence, num_frames):
    pairs_in_sequence = 0
    for x in range(1, num_frames + 1):
        track_ids = len(data_sequence.frame_to_gts[x])
        comparisons = int(((track_ids - 1) * (track_ids - 2)) / 2)
        pairs_in_sequence += comparisons
    return pairs_in_sequence

def calculate_overlapping_pairs(data_sequence, num_frames, threshold=0.5):
    frames = []
    num_overlapping_pairs = 0
    for x in range(1, num_frames + 1):
        track_ids = len(data_sequence.frame_to_gts[x])
        overlaps_in_frame = 0
        for i in range(0, track_ids):
            for j in range(i, track_ids):
                bbox1 = data_sequence.frame_to_gts[x][i].bbox
                bbox2 = data_sequence.frame_to_gts[x][j].bbox
                if calculate_overlap(bbox1, bbox2) > threshold:
                    overlaps_in_frame += 1
                    num_overlapping_pairs += 1  
    return num_overlapping_pairs

def calculate_overlap(bbox1, bbox2):
    bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    bbox2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
    xx1 = max(bbox1[0], bbox2[0])
    yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2])
    yy2 = min(bbox1[3], bbox2[3])
    if xx2 <= xx1 or yy2 <= yy1:
        return 0
    intersection = (xx2 - xx1) * (yy2 - yy1)
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0])\
        * (bbox2[3] - bbox2[1]) - intersection
    return intersection / union

def stats_to_json(num_sequences, num_frames_in_sequence, total_frames,
    num_track_ids_in_sequence, num_pairs_to_annotate_in_sequence,
    total_pairs_to_annotate, num_overlapping_pairs_in_sequence,
    total_overlapping_pairs, path):
    stats = {
        'num_sequences': num_sequences,
        'num_frames_in_sequence': num_frames_in_sequence,
        'total_frames': total_frames,
        'num_track_ids_in_sequence': num_track_ids_in_sequence,
        'num_pairs_to_annotate_in_sequence': num_pairs_to_annotate_in_sequence,
        'total_pairs_to_annotate': total_pairs_to_annotate,
        'num_overlapping_pairs_in_sequence': num_overlapping_pairs_in_sequence,
        'total_overlapping_pairs': total_overlapping_pairs
    }
    with open(path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    # dataset stat variables
    num_sequences = 0
    num_frames_in_sequence = [None]*100
    total_frames = 0
    num_track_ids_in_sequence = [None]*100
    num_pairs_to_annotate_in_sequence = [None]*100
    total_pairs_to_annotate = 0
    num_overlapping_pairs_in_sequence = [None]*100
    total_overlapping_pairs = 0
    for split in SPLITS:
        split_dir = os.path.join(DATA_PATH, split)
        seq_map_path = os.path.join(DATA_PATH, split + '_seqmap.txt')
        tracks, nums = get_track_names(seq_map_path)
        for track, num in zip(tracks, nums):
            seq_dir = os.path.join(split_dir, track)
            img_dir = os.path.join(seq_dir, 'img1')
            num_sequences += 1
            num_frames_in_sequence[num-1] = get_number_of_frames(img_dir)
            if split != 'test':
                data_sequence = Sequence(seq_dir)
                total_frames += num_frames_in_sequence[num-1]
                num_track_ids_in_sequence[num-1] = len(data_sequence.gt_track_id_to_gt_track)
                num_pairs_to_annotate_in_sequence[num-1] = get_num_pairs_to_annotate_in_sequence(
                    data_sequence, num_frames_in_sequence[num-1]
                )
                total_pairs_to_annotate += num_pairs_to_annotate_in_sequence[num-1]
                num_overlapping_pairs_in_sequence[num-1] = calculate_overlapping_pairs(
                    data_sequence, num_frames_in_sequence[num-1], THRESH
                )
                total_overlapping_pairs += num_overlapping_pairs_in_sequence[num-1]
    stats_to_json(num_sequences, num_frames_in_sequence, total_frames,
        num_track_ids_in_sequence, num_pairs_to_annotate_in_sequence,
        total_pairs_to_annotate, num_overlapping_pairs_in_sequence, 
        total_overlapping_pairs, f'dancetrack_stats/stats.json')

# TODO: Visualize data in matplotlib