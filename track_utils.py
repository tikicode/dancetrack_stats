import configparser
import os


class Detection:
    def __init__(self, bbox, frame, confidence=None, track_id=None):
        self.bbox = bbox  # ltwh
        self.frame = frame
        self.confidence = confidence
        self.track_id = track_id

        self.l, self.t, self.w, self.h = bbox
        self.r = self.l + self.w
        self.b = self.t + self.h
        self.cx = self.l + self.w/2
        self.cy = self.t + self.h/2

    def __repr__(self):
        fields = []
        fields.append(f'frame={self.frame:>4d}')
        l, t, w, h = self.bbox
        fields.append(f'bbox=({l:>6.1f} {t:>6.1f} {w:>6.1f} {h:>6.1f})')
        if self.confidence is not None:
            fields.append(f'conf={self.confidence:>5.3f}')
        if self.track_id is not None:
            fields.append(f'trid={self.track_id:>4d}')
        if hasattr(self, 'matched_id'):
            fields.append(f'mtid={self.matched_id:>4d}')
        repr_str = ', '.join(fields)
        repr_str = f'<Det: {repr_str}>'
        return repr_str


class Track:
    def __init__(self, dets, track_id):
        self.dets = dets
        self.track_id = track_id

        self.frame_to_det = {det.frame: det for det in dets}

        self.min_frame = min(det.frame for det in dets)
        self.max_frame = max(det.frame for det in dets)

    def get_det_by_frame(self, frame):
        if frame not in self.frame_to_det:
            return None
        return self.frame_to_det[frame]

    def __repr__(self):
        s = f'id={self.track_id:>4d}, '\
          + f'frames {self.min_frame:>4d} to {self.max_frame:>4d}'
        return f'< Track: {s} >'


class Sequence:
    def __init__(self, seq_dir):
        if not os.path.isdir(seq_dir):
            abs_path = os.path.abspath(seq_dir)
            raise ValueError(f"Directory '{abs_path}' does not exist.")

        self.dir = seq_dir

        self.frame_to_gts = {}
        self.gt_track_id_to_gt_track = {}
        self.frame_to_raw_dts = {}
        self.frame_to_tracked_dts = {}
        self.track_id_to_track = {}

        self.loaded_gts = False
        self.loaded_raw_dts = False
        self.loaded_tracked_dts = False

        seq_info_path = os.path.join(self.dir, 'seqinfo.ini')
        cfg = configparser.ConfigParser()
        cfg.read(seq_info_path)

        self.img_dir = cfg['Sequence'].get('imDir')
        self.img_dir = os.path.join(self.dir, self.img_dir)
        self.img_ext = cfg['Sequence'].get('imExt')
        self.seq_len = cfg['Sequence'].getint('seqLength')
        self.img_width = cfg['Sequence'].getint('imWidth')
        self.img_height = cfg['Sequence'].getint('imHeight')

        for fname in os.listdir(self.img_dir):
            if fname.endswith(self.img_ext):
                break
        l = len(os.path.splitext(fname)[0])
        self.img_path_fmt = os.path.join(self.img_dir,
            f'{{:>0{l}d}}{self.img_ext}')

        self.load_gt_tracks()

    def load_gt_tracks(self, path=None, cls_ids=(1,)):
        """Load ground-truth tracks."""
        self.frame_to_gts = {}
        for frame in range(1, self.seq_len + 1):
            self.frame_to_gts[frame] = []
        self.gt_track_id_to_gts = {}

        if path is None:
            path = os.path.join(self.dir, 'gt/gt.txt')
        with open(path) as f:
            for line in f:
                frame, track_id, l, t, w, h, cls, _, _ = line.split(',')
                if int(cls) not in cls_ids:
                    continue
                frame, track_id = map(int, (frame, track_id))
                l, t, w, h = map(float, (l, t, w, h))
                det = Detection(bbox=(l, t, w, h), frame=frame,
                    track_id=track_id)
                self.frame_to_gts[frame].append(det)
                self.gt_track_id_to_gts.setdefault(track_id, []).append(det)

        self.gt_track_id_to_gt_track = {}
        for track_id, dets in self.gt_track_id_to_gts.items():
            track = Track(dets=sorted(dets, key=lambda x: x.frame),
                track_id=track_id)
            self.gt_track_id_to_gt_track[track_id] = track

        self.loaded_gts = True

    def load_raw_detections(self, path=None):
        """Load tracker inputs."""
        self.frame_to_raw_dts = {}
        for frame in range(1, self.seq_len + 1):
            self.frame_to_raw_dts[frame] = []

        if path is None:
            path = os.path.join(self.dir, 'det/det.txt')
        with open(path) as f:
            for line in f:
                frame, _, l, t, w, h, conf = line.split(',')[:7]
                frame = int(frame)
                l, t, w, h, conf = map(float, (l, t, w, h, conf))
                det = Detection(bbox=(l, t, w, h), frame=frame,
                    confidence=conf)
                self.frame_to_raw_dts[frame].append(det)

        self.loaded_raw_dts = True

    def load_tracks(self, path):
        """Load tracker outputs."""
        self.frame_to_tracked_dts = {}
        for frame in range(1, self.seq_len + 1):
            self.frame_to_tracked_dts[frame] = []
        self.track_id_to_dts = {}

        with open(path) as f:
            for line in f:
                frame, track_id, l, t, w, h, conf = line.split(',')[:7]
                frame, track_id = map(int, (frame, track_id))
                l, t, w, h, conf = map(float, (l, t, w, h, conf))
                det = Detection(bbox=(l, t, w, h), frame=frame,
                    track_id=track_id, confidence=conf)
                self.frame_to_tracked_dts[frame].append(det)
                self.track_id_to_dts.setdefault(track_id, []).append(det)

        self.track_id_to_track = {}
        for track_id, dets in self.track_id_to_dts.items():
            track = Track(dets=sorted(dets, key=lambda x: x.frame),
                track_id=track_id)
            self.track_id_to_track[track_id] = track

        self.loaded_tracked_dts = True

    def get_gts_in_frame(self, frame):
        """Get ground-truth detections in frame."""
        return self.frame_to_gts[frame]

    def get_raw_dts_in_frame(self, frame, conf_thresh=None):
        """Get raw detections (before tracking) in frame."""
        if not self.loaded_raw_dts:
            raise RuntimeError('Please load raw detections first.')
        dets = self.frame_to_raw_dts[frame]
        if conf_thresh is not None:
            dets = [d for d in dets if d.confidence > conf_thresh]
        return dets

    def get_dts_in_frame(self, frame):
        """Get tracked detections in frame."""
        if not self.loaded_tracked_dts:
            raise RuntimeError('Please load tracked detections first.')
        return self.frame_to_tracked_dts[frame]

    def get_frame_img_path(self, frame):
        """Get full image path by frame id."""
        return self.img_path_fmt.format(frame)

    def get_gt_tracks(self, track_ids=None):
        """Get ground-truth tracks in sequence."""
        if track_ids is None:
            return list(self.gt_track_id_to_gt_track.values())
        if not isinstance(track_ids, (list, tuple)):
            track_ids = [track_ids]
        return [self.gt_track_id_to_gt_track[track_id]
            for track_id in track_ids]

    def get_tracks(self, track_ids=None):
        """Get tracker output tracks in sequence."""
        if track_ids is None:
            return list(self.track_id_to_track.values())
        if not isinstance(track_ids, (list, tuple)):
            track_ids = [track_ids]
        return [self.track_id_to_track[track_id]
            for track_id in track_ids]