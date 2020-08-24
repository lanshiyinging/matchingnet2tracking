from filterpy.kalman import KalmanFilter
import numpy as np
import lap

def iou(a, b, criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """

    x1 = np.maximum(a[0], b[0])
    y1 = np.maximum(a[1], b[1])
    x2 = np.minimum(a[2], b[2])
    y2 = np.minimum(a[3], b[3])

    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)

    inter = w * h
    aarea = (a[2] - a[0]) * (a[3] - a[1])
    barea = (b[2] - b[0]) * (b[3] - b[1])
    # intersection over union overlap
    if criterion.lower() == "union":
        o = inter / float(aarea + barea - inter)
    elif criterion.lower() == "a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_det_to_trk(dets, trks, iou_threshold=0.3):
    if(len(trks) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5))
    iou_matrix = np.zeros((len(dets), len(trks)))

    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trks):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
            


class Track:
    def __init__(self, bbox, track_id, frame_id):
        '''
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        '''
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.

        self.id = track_id
        self.time_since_update = 0
        self.history = [[frame_id, x, y]]
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):

        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def convert_bbox_to_z(self, bbox):
        # convert [x1,y1,x2,y2] to [x,y,s,r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)

        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        # convert [x, y, s, r] to [x1, y1, x2, y2]
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if (score == None):
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

class Tracker:
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0

    def update(self, gallery_images, gallery_labels, query_images, query_labels, num_class,  x_batch, y_batch, frame_batch, d):

        self.frame_count += 1
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        
        # 构造x_batch, y_batch, frame_batch
        # 送入网络，获得结果
        # 根据分类结果，获得matched
        # 处理unmatched trk
        # 处理unmatched det
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.tracks.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_det_to_trk(dets, trks)

        for m in matched:
            self.tracks[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = Track(dets[i, :], self.next_id)
            self.tracks.append(trk)
            self.next_id += 1

        for i, trk in enumerate(self.tracks):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

            #i += 1
            if(trk.time_since_update > self.max_age):
                self.tracks.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 5))




