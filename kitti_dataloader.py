import os
import numpy as np
import lap
import random
import cv2
import torch
import preprocessing

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

class KittiDataLoader():
    def __init__(self, data_root_path, image_root, annotation_root, det_root, training_set, validation_set, testing_set, seq_num, transform=None):
        self.data_root_path = data_root_path
        self.image_root_path = os.path.join(self.data_root_path, image_root)
        self.annotation_root_path = os.path.join(self.data_root_path, annotation_root)
        self.det_root_path = os.path.join(self.data_root_path, det_root)
        self.transform = transform
        self.seq_num = seq_num
        self.det_dict = self.get_detections()
        self.gt_dict = self.get_annotations()
        self.get_TD_label()
        self.dataset_split = {'train': training_set, 'val': validation_set, 'test': testing_set}
        self.index = {'train': 0, 'val': 0, 'test': 0}
        self.datasets = {'train': self.generate_samples('train'),
                         'val': self.generate_samples('val'),
                         'test': self.generate_samples('test')}


    
    def get_detections(self):
        det_dict = {x:{} for x in range(self.seq_num)}
        det_files = os.listdir(self.det_root_path)
        det_files.sort()
        for det_fn in det_files:
            seq = int(det_fn.split('.')[0])
            det_path = os.path.join(self.det_root_path, det_fn)
            dets = open(det_path).readlines()
            dets_anno = [det.strip('\r\n').strip('\n').split(',') for det in dets]
            last_frame = int(dets_anno[-1][0])
            seq_dets = {x:[] for x in range(last_frame+1)}
            for det_ann in dets_anno:
                frame_id = int(det_ann[0])
                conf = float(det_ann[6])
                if conf < 0.5:
                    continue
                bbox = list(map(float, det_ann[2:6]))
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                #ann = [bbox, -1]
                bbox.extend([-1])
                seq_dets[frame_id].append(bbox)
            '''
            for frame_det in seq_dets:
                bbox = [d[:4] for d in frame_det]
                for box in bbox:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                boxes = np.array(bbox)
                indices = preprocessing.non_max_suppression(boxes, 0.5)
                frame_det = [frame_det[i] for i in indices]
            '''
            det_dict[seq] = seq_dets
        
        return det_dict


    def get_annotations(self):
        gt_dict = {x:{} for x in range(self.seq_num)}
        gt_files = os.listdir(self.annotation_root_path)
        gt_files.sort()
        for gt_fn in gt_files:
            seq = int(gt_fn.split('.')[0])
            gt_path = os.path.join(self.annotation_root_path, gt_fn)
            gts = open(gt_path).readlines()
            gt_annos = [gt.strip('\r\n').strip('\n').split(' ') for gt in gts]
            last_frame = int(gt_annos[-1][0])
            seq_gt = {x: [] for x in range(last_frame+1)}
            for gt_ann in gt_annos:
                if gt_ann[2] == 'Car' or gt_ann[2] == 'Van':
                    frame_id = int(gt_ann[0])
                    id = int(gt_ann[1])
                    bbox = list(map(float, gt_ann[6:10]))
                    bbox.extend([id])
                    seq_gt[frame_id].append(bbox)
            for _, frame_gt in seq_gt.items():
                bbox = [d[:4] for d in frame_gt]
                for box in bbox:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                boxes = np.array(bbox)
                indices = preprocessing.non_max_suppression(boxes, 0.5)
                frame_gt = [frame_gt[i] for i in indices]
            gt_dict[seq] = seq_gt
        return gt_dict

    def get_TD_label(self):
        # if matched with gt, assign track_id
        for seq, seq_dets in self.det_dict.items():
            seq_gt = self.gt_dict[seq]
            for frame, frame_det in seq_dets.items():
                frame_gt = seq_gt[frame]
                frame_det_np = np.array(frame_det)
                frame_gt_np = np.array(frame_gt)
                matched, unmatched_dets, unmatched_gts = associate_det_to_trk(frame_det_np, frame_gt_np)
                for m in matched:
                    id = int(frame_gt_np[m[1], -1])
                    frame_det[m[0]][-1] = id

    def generate_samples(self, data_type):
        all_samples = []
        seq_list = self.dataset_split[data_type]
        random.shuffle(seq_list)
        for seq in seq_list:
            seq_dets = self.det_dict[seq]
            frame_list = list(seq_dets.keys())
            for i in range(len(frame_list)-1):
                gallery_images = []
                gallery_labels = []
                query_images = []
                query_labels = []
                r_gallery_images = []
                r_gallery_labels = []
                r_query_images = []
                r_query_labels = []

                frame_g = frame_list[i+1]
                frame_q = frame_list[i]

                img_g_path = os.path.join(self.image_root_path, '%04d' %seq, '%06d.png' %frame_g)
                img_q_path = os.path.join(self.image_root_path, '%04d' %seq, '%06d.png' %frame_q)

                img_g = cv2.imread(img_g_path)
                img_q = cv2.imread(img_q_path)

                gallery_dets = seq_dets[frame_g]
                query_dets = seq_dets[frame_q]

                if len(gallery_dets) == 0 or len(query_dets) == 0:
                    continue

                id2label = {}
                j = 0
                for det in gallery_dets:
                    bbox = list(map(int, det[:4]))
                    id = det[4]
                    det_img = img_g[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    if self.transform:
                        det_img = self.transform(det_img)
                    if id == -1:
                        id2label[id] = j
                    label = j
                    j += 1
                    gallery_images.append(det_img)
                    gallery_labels.append(label)

                for det in query_dets:
                    bbox = list(map(int, det[:4]))
                    id = det[4]
                    det_img = img_q[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    if self.transform:
                        det_img = self.transform(det_img)
                    if id != -1:
                        if id in id2label:
                            label = id2label[id]
                    else:
                        continue
                    query_images.append(det_img)
                    query_labels.append(label)

                r_j = 0
                r_id2label = {}
                for det in query_dets:
                    bbox = list(map(int, det[:4]))
                    id = det[4]
                    det_img = img_q[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    if self.transform:
                        det_img = self.transform(det_img)
                    if id == -1:
                        id2label[id] = r_j
                    label = r_j
                    r_j += 1
                    r_gallery_images.append(det_img)
                    r_gallery_labels.append(label)

                for det in gallery_dets:
                    bbox = list(map(int, det[:4]))
                    id = det[4]
                    det_img = img_g[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    if self.transform:
                        det_img = self.transform(det_img)
                    if id != -1:
                        if id in r_id2label:
                            label = r_id2label[id]
                    else:
                        continue
                    r_query_images.append(det_img)
                    r_query_labels.append(label)
                
                if len(gallery_images) == 0 or len(r_gallery_images) == 0 or len(query_images) == 0 or len(r_query_images) == 0:
                    continue

                all_samples.append([gallery_images, gallery_labels, query_images, query_labels, j, r_gallery_images, r_gallery_labels, r_query_images, r_query_labels, r_j])
        random.shuffle(all_samples)
        return all_samples

    def get_iter_num(self, data_type):
        return len(self.datasets[data_type])

    def get_batch(self, data_type):
        all_samples = self.datasets[data_type]
        if self.index[data_type] >= len(all_samples):
            self.index[data_type] = 0
        ind = self.index[data_type]
        gallery_images, gallery_labels, query_images, query_labels, num_class, r_gallery_images, r_gallery_labels, r_query_images, r_query_labels, r_num_class = all_samples[ind]
        self.index[data_type] += 1
        return np.array(gallery_images), np.array(gallery_labels), np.array(query_images), np.array(query_labels), num_class, np.array(r_gallery_images), np.array(r_gallery_labels), np.array(r_query_images), np.array(r_query_labels), r_num_class


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image):
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        return new_image

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, image):

        return ((image.astype(np.float32) - self.mean) / self.std)

def test():
    KittiDataLoader(data_root_path='data', image_root='image_02', 
                           annotation_root='label_02', det_root='det_02', 
                           training_set=[1,2,3], validation_set=[4,5,6], 
                           testing_set=[0], seq_num=21)


if __name__ == "__main__":
    test()