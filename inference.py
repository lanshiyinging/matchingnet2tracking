import os
import torch
import argparse
import yaml
import cv2
import view
import numpy as np
import time
import matplotlib.pyplot as plt
from Efficientdet.backbone import EfficientDetBackbone
from Efficientdet.efficientdet.utils import BBoxTransform, ClipBoxes
from Efficientdet.utils.utils import preprocess, invert_affine, postprocess
from Tracker import Track, Tracker
from matchingnet_SA import MatchingNet

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project', type=str, default='kitti')
parser.add_argument('-s', '--sequence', type=str, default='0000')
parser.add_argument('-w', '--weights', type=str, default=None)
parser.add_argument('-sa_w', '--sa_weights', type=str, default=None)
parser.add_argument('--display', type=bool, default=True)
# RNN size
parser.add_argument('--human_node_rnn_size', type=int, default=128, help='Size of Human Node RNN hidden state')
parser.add_argument('--human_human_edge_rnn_size', type=int, default=256, help='Size of Human Human Edge RNN hidden state')

# Input and output size
parser.add_argument('--human_node_input_size', type=int, default=2, help='Dimension of the node features')
parser.add_argument('--human_human_edge_input_size', type=int, default=2, help='Dimension of the edge features')
parser.add_argument('--human_node_output_size', type=int, default=5, help='Dimension of the node output')

# Embedding size
parser.add_argument('--human_node_embedding_size', type=int, default=64, help='Embedding size of node features')
parser.add_argument('--human_human_edge_embedding_size', type=int, default=64, help='Embedding size of edge features')

# Attention vector dimension
parser.add_argument('--attention_size', type=int, default=64, help='Attention size')

# Sequence length
parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')
parser.add_argument('--pred_length', type=int, default=12, help='Predicted sequence length')

parser.add_argument('--feat_dim', type=int, default=512, help='embedding dimention')
parser.add_argument('--epoch_num', type=int, default=100, help='training epoch')
parser.add_argument('--log_dir', type=str, default='log', help='log dir')
parser.add_argument('--data_dir', type=str, default='data', help='dataset dir')
parser.add_argument('--save_dir', type=str, default='model', help='model dir')
parser.add_argument('--seq_num', type=int, default=21, help='the total number of sequences')

parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
args = parser.parse_args()

compound_coef = 6
nms_threshold = 0.5
use_cuda = args.use_cuda
use_float16 = False
project_name = args.project
weights_path = 'Efficientdet/logs/kitti_coco_d6_4class/efficientdet-d6_85_64328.pth' if args.weights is None else args.weights
SA_weights_path = args.sa_weights

params = yaml.safe_load(open('Efficientdet/projects/%s.yml' %project_name))
obj_list = params['obj_list']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def detector(model, image_path, threshold=0.5):

    detections = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
    x = torch.from_numpy(framed_imgs[0])

    if use_cuda:
        x = x.cuda()
        if use_float16:
            x = x.half()
        else:
            x = x.float()
    else:
        x = x.float()

    x = x.unsqueeze(0).permute(0, 3, 1, 2)
    features, regression, classification, anchors = model(x)
    preds = postprocess(x, anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, nms_threshold)

    if not preds:
        return None

    preds = invert_affine(framed_metas, preds)[0]

    scores = preds['scores']
    class_ids = preds['class_ids']
    rois = preds['rois']

    if rois.shape[0] > 0:
        # x1,y1,x2,y2 -> x1,y1,w,h
        rois[:, 2] -= rois[:, 0]
        rois[:, 3] -= rois[:, 1]

        bbox_score = scores

        for roi_id in range(rois.shape[0]):
            score = float(bbox_score[roi_id])
            label = int(class_ids[roi_id])
            box = rois[roi_id, :]

            if label == 0:
                image_result = {
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                detections.append(image_result)

    if not len(detections):
        #raise Exception('invalid output')
        return None

    return detections

def main():

    seq_root_dir = 'data/kitti/training/image_02/'
    out_root_dir = 'output/training/'

    # load detector
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    SA_model = MatchingNet(args)
    SA_model.load_state_dict(torch.load(SA_weights_path, map_location=torch.device('cpu')))
    SA_model.requires_grad_(False)
    SA_model.eval()

    if use_cuda:
        model.cuda()
        SA_model.cuda()

        if use_float16:
            model.half()

    for seq in os.listdir(seq_root_dir):
        if args.sequence != 'all':
            if args.sequence not in seq:
                continue
        seq_dir = seq_root_dir + seq + '/'
        out_path = out_root_dir + seq + '.txt'
        f = open(out_path, 'w')
        det_f = open('output/det/%s.txt' % seq, 'w')
        tracker = Tracker()
        
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('output/test/%s.avi' % seq, fourcc, 10.0, (1280, 375))

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output/test/%s.mp4' % seq, fourcc, 10.0, (1280, 375))
        fn_list = os.listdir(seq_dir)
        fn_list.sort()
        total_det_time = 0
        total_track_time = 0
        total_frames = 0
        for fn in fn_list:
            total_frames += 1
            frame_id = int(fn.split('.')[0])
            print("Processing sequence %s:%d" % (seq, frame_id))
            frame_path = seq_dir + fn
            det_start_time = time.time()
            detections = detector(model, frame_path)
            det_time = time.time() - det_start_time
            total_det_time += det_time
            if detections is not None:
                dets = [x['bbox'] for x in detections]
                dets[:, 2:4] += dets[:, 0:2]
            else:
                dets=np.empty((0, 4))

            for det in dets:
                det_f.write("%d %f %f %f %f\n" % (frame_id, det[0], det[1], det[2], det[3]))

            img = cv2.imread(frame_path)

            track_start_time = time.time()
            results = tracker.update(SA_model, img, dets)
            track_time = time.time() - track_start_time
            total_track_time += track_time

            
            for ret in results:
                f.write("%d %d Car -1 -1 -1 %f %f %f %f -1 -1 -1 -1 -1 -1 -1 -1\n" % (frame_id, ret[4], ret[0], ret[1], ret[2], ret[3]))
                view.rectangle(img, ret[0], ret[1], ret[2]-ret[0], ret[3]-ret[1], str(int(ret[4])))
            out.write(cv2.resize(img, (1280, 375)))
            if args.display:
                cv2.imshow(seq, cv2.resize(img, (1280, 375)))
                cv2.waitKey(100)
        out.release()
        cv2.destroyAllWindows()

        print("Total Detection took: %.3f seconds for %d frames or %.1f FPS" % (total_det_time, total_frames, total_frames / total_det_time))
        print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_track_time, total_frames, total_frames / total_track_time))


if __name__ == '__main__':
    main()




