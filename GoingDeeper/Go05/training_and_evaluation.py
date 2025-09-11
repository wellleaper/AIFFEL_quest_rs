"""
This script combines the entire object detection pipeline into a single file for training and evaluation.
It includes:
- Configuration management
- Data loading from TFRecord files
- RetinaNet model definition with a ResNet backbone
- FPN (Feature Pyramid Network)
- Custom loss functions (Focal Loss, Smooth L1)
- Label encoding (matching ground truth to anchor boxes)
- A training loop with checkpointing and validation
- An evaluation loop to assess model performance on a test set.

├── data
│   ├── checkpoints -> /home/jovyan/data/checkpoints
│   ├── downloads -> /home/jovyan/data/downloads
│   ├── go_1.png -> /home/jovyan/data/go_1.png
│   ├── go_2.png -> /home/jovyan/data/go_2.png
│   ├── go_3.png -> /home/jovyan/data/go_3.png
│   ├── go_4.png -> /home/jovyan/data/go_4.png
│   ├── go_5.png -> /home/jovyan/data/go_5.png
│   ├── kitti -> /home/jovyan/data/kitti
│   ├── stop_1.png -> /home/jovyan/data/stop_1.png
│   ├── stop_2.png -> /home/jovyan/data/stop_2.png
│   ├── stop_3.png -> /home/jovyan/data/stop_3.png
│   ├── stop_4.png -> /home/jovyan/data/stop_4.png
│   └── stop_5.png -> /home/jovyan/data/stop_5.png
├── training_and_evaluation.py
└── test_loss.py


To run training:
python training_and_evaluation.py train

To run evaluation on all saved checkpoints:
python training_and_evaluation.py evaluate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.ops import nms
from tfrecord.torch.dataset import TFRecordDataset
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import glob
import re
import io
import math
import itertools
import copy

# =================================================================================
# 1. CONFIGURATION
# =================================================================================

class TrainingConfig:
    # The root directory of the project, assuming it's in a fixed location relative to HOME
    ROOT_DIR = os.path.join(os.getenv("HOME"), "work/object_detection")

    # Data
    DATA_PATH = os.path.join(ROOT_DIR, "data")
    TRAIN_TFRECORD_PATTERN = os.path.join(ROOT_DIR, "data/kitti/3.2.0/kitti-train.tfrecord-*")
    VALIDATION_TFRECORD_PATTERN = os.path.join(ROOT_DIR, "data/kitti/3.2.0/kitti-validation.tfrecord-*")
    TEST_TFRECORD_PATTERN = os.path.join(ROOT_DIR, "data/kitti/3.2.0/kitti-test.tfrecord-*")

    # Model
    NUM_CLASSES = 8

    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATES = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    LEARNING_RATE_BOUNDARIES = [125, 250, 500, 240000, 360000]
    OPTIMIZER_MOMENTUM = 0.9

    # Loss parameters
    CLASSIFICATION_LOSS_ALPHA = 0.25
    CLASSIFICATION_LOSS_GAMMA = 2.0
    BOX_LOSS_DELTA = 1.0

    # Decoder parameters
    CONFIDENCE_THRESHOLD = 0.05
    NMS_IOU_THRESHOLD = 0.5
    MAX_DETECTIONS_PER_CLASS = 100
    MAX_DETECTIONS = 100
    BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]

    # Checkpoint
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'data', 'checkpoints')
    LATEST_CHECKPOINT_FILE = os.path.join(CHECKPOINT_PATH, 'latest_checkpoint.pth')
    BEST_CHECKPOINT_FILE = os.path.join(CHECKPOINT_PATH, 'best_checkpoint.pth')
    EPOCH_CHECKPOINT_TEMPLATE = os.path.join(CHECKPOINT_PATH, 'checkpoint_epoch_{epoch}.pth')

config = TrainingConfig()


# =================================================================================
# 2. UTILITY FUNCTIONS & CLASSES
# =================================================================================

# --- From box_utils.py ---

def swap_xy(boxes):
    return torch.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], dim=-1)

def convert_to_xywh(boxes):
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0,
         boxes[..., 2:] - boxes[..., :2]],
        dim=-1
    )

def convert_to_corners(boxes):
    return torch.cat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        dim=-1,
    )

def compute_iou(boxes1, boxes2):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = torch.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = torch.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = torch.maximum(torch.tensor(0.0), rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = torch.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, torch.tensor(1e-8)
    )
    return torch.clamp(intersection_area / union_area, min=0.0, max=1.0)

# --- From anchor.py ---

class AnchorBox:
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []
        for area in self._areas:
            all_dims_for_area = []
            for ratio in self.aspect_ratios:
                anchor_height = torch.tensor(area / ratio).sqrt()
                anchor_width = area / anchor_height
                for scale in self.scales:
                    w = anchor_width * scale
                    h = anchor_height * scale
                    all_dims_for_area.append(torch.tensor([w, h]))
            stacked_dims = torch.stack(all_dims_for_area, dim=0)
            reshaped_dims = stacked_dims.view(1, 1, self._num_anchors, 2)
            anchor_dims_all.append(reshaped_dims)
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        rx = torch.arange(feature_width, dtype=torch.float32) + 0.5
        ry = torch.arange(feature_height, dtype=torch.float32) + 0.5
        centers = torch.stack(torch.meshgrid(rx, ry, indexing='xy'), dim=-1) * self._strides[level - 3]
        centers = centers.unsqueeze(-2)
        centers = centers.repeat(1, 1, self._num_anchors, 1)
        dims = self._anchor_dims[level - 3].repeat(feature_height, feature_width, 1, 1)
        anchors = torch.cat([centers, dims], dim=-1)
        return anchors.view(feature_height * feature_width * self._num_anchors, 4)

    def get_anchors(self, image_height, image_width):
        anchors = [
            self._get_anchors(
                math.ceil(image_height / 2 ** i),
                math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return torch.cat(anchors, dim=0)

# --- From label_encoder.py ---

class LabelEncoder:
    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = torch.tensor(config.BOX_VARIANCE, dtype=torch.float32)

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou, _ = torch.max(iou_matrix, dim=1)
        matched_gt_idx = torch.argmax(iou_matrix, dim=1)
        positive_mask = max_iou >= match_iou
        negative_mask = max_iou < ignore_iou
        ignore_mask = ~(positive_mask | negative_mask)
        return matched_gt_idx, positive_mask.float(), ignore_mask.float()

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = torch.cat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                torch.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            dim=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        anchor_boxes = self._anchor_box.get_anchors(image_shape[2], image_shape[3])
        cls_ids = cls_ids.float()
        gt_boxes = convert_to_xywh(gt_boxes)
        if gt_boxes.numel() == 0:
            box_target = torch.zeros_like(anchor_boxes)
            cls_target = torch.full((anchor_boxes.shape[0],), -1.0, dtype=torch.float32)
            return box_target, cls_target

        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        matched_gt_boxes = torch.gather(gt_boxes, 0, matched_gt_idx.unsqueeze(1).repeat(1, gt_boxes.shape[1]))
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = torch.gather(cls_ids, 0, matched_gt_idx)
        
        cls_target = torch.where(positive_mask != 1.0, -1.0, matched_gt_cls_ids)
        cls_target = torch.where(ignore_mask == 1.0, -2.0, cls_target)
        return box_target, cls_target

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        box_targets = []
        cls_targets = []
        for i in range(batch_images.size(0)):
            box_target, cls_target = self._encode_sample(batch_images.size(), gt_boxes[i], cls_ids[i])
            box_targets.append(box_target)
            cls_targets.append(cls_target)
        
        box_targets = torch.stack(box_targets, dim=0)
        cls_targets = torch.stack(cls_targets, dim=0)
        
        # Normalization is applied here
        normalized_images = (batch_images - 0.485) / 0.229
        normalized_images = (normalized_images - 0.456) / 0.224
        normalized_images = (normalized_images - 0.406) / 0.225

        return normalized_images, (cls_targets, box_targets)

# --- From decoder.py ---

class DecodePredictions(nn.Module):
    def __init__(self):
        super(DecodePredictions, self).__init__()
        self.num_classes = config.NUM_CLASSES
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.nms_iou_threshold = config.NMS_IOU_THRESHOLD
        self.max_detections_per_class = config.MAX_DETECTIONS_PER_CLASS
        self.max_detections = config.MAX_DETECTIONS
        self._anchor_box = AnchorBox()
        self._box_variance = torch.tensor(config.BOX_VARIANCE, dtype=torch.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = torch.cat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                torch.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            dim=-1,
        )
        return convert_to_corners(boxes)

    def forward(self, images, predictions):
        image_shape = images.shape
        anchor_boxes = self._anchor_box.get_anchors(image_shape[2], image_shape[3])
        box_predictions = predictions[:, :, :4]
        cls_predictions = torch.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        
        boxes = boxes.squeeze(0)
        scores = cls_predictions.squeeze(0)

        all_boxes, all_scores, all_classes = [], [], []

        for class_id in range(self.num_classes):
            class_scores = scores[:, class_id]
            mask = class_scores > self.confidence_threshold
            if not mask.any():
                continue

            class_boxes = boxes[mask]
            class_scores = class_scores[mask]

            if len(class_scores) > self.max_detections_per_class:
                top_k = torch.topk(class_scores, self.max_detections_per_class)
                class_scores, class_boxes = top_k.values, class_boxes[top_k.indices]

            keep = nms(class_boxes, class_scores, self.nms_iou_threshold)
            all_boxes.append(class_boxes[keep])
            all_scores.append(class_scores[keep])
            all_classes.append(torch.full_like(class_scores[keep], class_id, dtype=torch.int64))

        if not all_boxes:
            return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)

        final_boxes = torch.cat(all_boxes, dim=0)
        final_scores = torch.cat(all_scores, dim=0)
        final_classes = torch.cat(all_classes, dim=0)

        if len(final_scores) > self.max_detections:
            overall_top_k = torch.topk(final_scores, self.max_detections)
            final_scores, final_boxes, final_classes = overall_top_k.values, final_boxes[overall_top_k.indices], final_classes[overall_top_k.indices]

        return final_boxes, final_scores, final_classes

# --- From visualization.py ---

def visualize_detections(image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        origin_x, origin_y = x1, y1 
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([origin_x, origin_y], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(origin_x, origin_y, text, bbox={"facecolor": color, "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True)
    plt.show()
    return ax


# =================================================================================
# 3. DATA LOADING
# =================================================================================

class TFRecordKITTIDataset(IterableDataset):
    def __init__(self, tfrecord_pattern, transform=None):
        super(TFRecordKITTIDataset, self).__init__()
        self.user_transform = transform
        self.file_paths = sorted([p for p in glob.glob(tfrecord_pattern) if not p.endswith('.index')])
        if not self.file_paths:
            raise FileNotFoundError(f"No TFRecord files found for pattern: {tfrecord_pattern}")

        feature_description = {
            'image': "byte", 'image/file_name': "byte",
            'objects/bbox': "float", 'objects/type': "int",
        }
        
        self.datasets = [
            TFRecordDataset(path, index_path=None, description=feature_description, transform=self._parse_record)
            for path in self.file_paths
        ]
        self._total_size = None

    def _parse_record(self, features):
        image = Image.open(io.BytesIO(features['image'])).convert('RGB')
        filename = features['image/file_name'].decode('utf-8')
        
        if 'objects/bbox' in features and 'objects/type' in features:
            bboxes = torch.tensor(features['objects/bbox'], dtype=torch.float32).view(-1, 4)
            bboxes = bboxes[:, [1, 0, 3, 2]] # y_min, x_min, y_max, x_max -> x_min, y_min, x_max, y_max
            labels = torch.tensor(features['objects/type'], dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        sample = {'image': image, 'bbox': bboxes, 'class_id': labels, 'filename': filename}
        
        if self.user_transform:
            sample['image'] = self.user_transform(sample['image'])
            
        return sample

    def __iter__(self):
        return itertools.chain.from_iterable(self.datasets)

    def __len__(self):
        if self._total_size is None:
            self._total_size = sum(1 for _ in self)
        return self._total_size

def collate_fn(batch):
    images, bboxes, class_ids = zip(*[(item['image'], item['bbox'], item['class_id']) for item in batch])
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        padding = (0, max_w - w, 0, max_h - h)
        padded_img = F.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
    
    return torch.stack(padded_images, 0), list(bboxes), list(class_ids)


# =================================================================================
# 4. MODEL DEFINITION (RETINANET)
# =================================================================================

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, images):
        c1 = self.layer1(images)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2, c3, c4

class FeaturePyramid(nn.Module):
    def __init__(self, backbone):
        super(FeaturePyramid, self).__init__()
        self.backbone = backbone
        self.conv_c3_1x1 = nn.Conv2d(512, 256, 1, 1, padding=0)
        self.conv_c4_1x1 = nn.Conv2d(1024, 256, 1, 1, padding=0)
        self.conv_c5_1x1 = nn.Conv2d(2048, 256, 1, 1, padding=0)
        self.conv_c3_3x3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv_c4_3x3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv_c5_3x3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv_c6_3x3 = nn.Conv2d(2048, 256, 3, 2, padding=1)
        self.conv_c7_3x3 = nn.Conv2d(256, 256, 3, 2, padding=1)

    def forward(self, images):
        c3_output, c4_output, c5_output = self.backbone(images)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_lat = self.conv_c4_1x1(c4_output)
        p5_upsampled = F.interpolate(p5_output, size=p4_lat.shape[2:], mode='nearest')
        p4_output = p4_lat + p5_upsampled
        p3_lat = self.conv_c3_1x1(c3_output)
        p4_upsampled = F.interpolate(p4_output, size=p3_lat.shape[2:], mode='nearest')
        p3_output = p3_lat + p4_upsampled
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(F.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output

def build_head(output_filters, bias_init):
    layers = []
    for _ in range(4):
        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    final_conv = nn.Conv2d(256, output_filters, kernel_size=3, stride=1, padding=1)
    if isinstance(bias_init, float):
        torch.nn.init.constant_(final_conv.bias, bias_init)
    else: # zeros
        torch.nn.init.constant_(final_conv.bias, 0.0)
    layers.append(final_conv)
    return nn.Sequential(*layers)

class RetinaNet(nn.Module):
    def __init__(self, num_classes, backbone):
        super(RetinaNet, self).__init__()
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        prior_probability = -torch.log(torch.tensor((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def forward(self, image):
        features = self.fpn(image)
        N = image.size(0)
        cls_outputs, box_outputs = [], []
        for feature in features:
            box_outputs.append(self.box_head(feature).view(N, -1, 4))
            cls_outputs.append(self.cls_head(feature).view(N, -1, self.num_classes))
        return torch.cat(cls_outputs, dim=1), torch.cat(box_outputs, dim=1)

def get_backbone():
    return ResNetBackbone()


# =================================================================================
# 5. LOSS FUNCTION
# =================================================================================

class RetinaNetBoxLoss(nn.Module):
    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__()
        self._delta = delta

    def forward(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = torch.abs(difference)
        loss = torch.where(absolute_difference < self._delta, 0.5 * (difference ** 2), absolute_difference - 0.5)
        return torch.sum(loss, dim=-1)

class RetinaNetClassificationLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, y_true, y_pred):
        cross_entropy = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        probs = torch.sigmoid(y_pred)
        alpha = torch.where(y_true == 1.0, self._alpha, 1.0 - self._alpha)
        pt = torch.where(y_true == 1.0, probs, 1.0 - probs)
        loss = alpha * torch.pow(1.0 - pt, self._gamma) * cross_entropy
        return torch.sum(loss, dim=-1)

class RetinaNetLoss(nn.Module):
    def __init__(self):
        super(RetinaNetLoss, self).__init__()
        self._clf_loss = RetinaNetClassificationLoss(config.CLASSIFICATION_LOSS_ALPHA, config.CLASSIFICATION_LOSS_GAMMA)
        self._box_loss = RetinaNetBoxLoss(config.BOX_LOSS_DELTA)
        self._num_classes = config.NUM_CLASSES

    def forward(self, y_pred, y_true):
        cls_preds, box_preds = y_pred
        cls_targets, box_targets = y_true
        
        positive_mask = (cls_targets > -1.0).float()
        ignore_mask = (cls_targets == -2.0).float()
        normalizer = torch.sum(positive_mask, dim=-1).clamp(min=1.0)

        temp_cls_targets = torch.where(cls_targets >= 0, cls_targets.long(), 0)
        one_hot_labels = F.one_hot(temp_cls_targets, num_classes=self._num_classes).float()
        one_hot_labels = one_hot_labels * positive_mask.unsqueeze(-1)

        clf_loss = self._clf_loss(one_hot_labels, cls_preds.float())
        clf_loss = torch.where(ignore_mask == 1.0, 0.0, clf_loss)
        clf_loss = torch.sum(clf_loss, dim=-1) / normalizer

        box_loss = self._box_loss(box_targets.float(), box_preds.float())
        box_loss = torch.where(positive_mask == 1.0, box_loss, 0.0)
        box_loss = torch.sum(box_loss, dim=-1) / normalizer

        return clf_loss + box_loss


# =================================================================================
# 6. MAIN LOGIC: TRAINING & EVALUATION
# =================================================================================

def run_training():
    print("\n--- Starting Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    pil_to_tensor = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = TFRecordKITTIDataset(config.TRAIN_TFRECORD_PATTERN, transform=pil_to_tensor)
    validation_dataset = TFRecordKITTIDataset(config.VALIDATION_TFRECORD_PATTERN, transform=pil_to_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    print("\n--- Dataset and DataLoader Info ---")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Number of training batches per epoch: {len(train_loader)}")
    print(f"Number of validation batches per epoch: {len(validation_loader)}")
    print("-------------------------------------\
")

    model = RetinaNet(config.NUM_CLASSES, get_backbone()).to(device)
    loss_fn = RetinaNetLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATES[0], momentum=config.OPTIMIZER_MOMENTUM)
    
    def lr_lambda(epoch):
        for i, boundary in enumerate(config.LEARNING_RATE_BOUNDARIES):
            if epoch < boundary: return config.LEARNING_RATES[i] / config.LEARNING_RATES[0]
        return config.LEARNING_RATES[-1] / config.LEARNING_RATES[0]
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    label_encoder = LabelEncoder()

    start_epoch, best_val_loss = 0, float('inf')
    if os.path.exists(config.LATEST_CHECKPOINT_FILE):
        checkpoint = torch.load(config.LATEST_CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        for i, (batch_images, gt_boxes, cls_ids) in enumerate(train_loader):
            encoded_images, encoded_labels = label_encoder.encode_batch(batch_images, gt_boxes, cls_ids)
            encoded_images = encoded_images.to(device)
            cls_targets, box_targets = encoded_labels[0].to(device), encoded_labels[1].to(device)
            
            predictions = model(encoded_images)
            loss = torch.mean(loss_fn(predictions, (cls_targets, box_targets)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_images, gt_boxes, cls_ids in validation_loader:
                encoded_images, encoded_labels = label_encoder.encode_batch(batch_images, gt_boxes, cls_ids)
                encoded_images = encoded_images.to(device)
                cls_targets, box_targets = encoded_labels[0].to(device), encoded_labels[1].to(device)
                predictions = model(encoded_images)
                loss = torch.mean(loss_fn(predictions, (cls_targets, box_targets)))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(validation_loader)
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Duration: {time.time() - start_time:.1f}s\n")

        scheduler.step()

        # Save latest checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config.LATEST_CHECKPOINT_FILE)
        
        # Save epoch-specific checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, config.EPOCH_CHECKPOINT_TEMPLATE.format(epoch=epoch+1))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, config.BEST_CHECKPOINT_FILE)
            print(f"*** New best model saved with validation loss: {avg_val_loss:.4f} ***\n")

def run_evaluation():
    print("\n--- Starting Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pil_to_tensor = transforms.Compose([transforms.ToTensor()])
    test_dataset = TFRecordKITTIDataset(config.TEST_TFRECORD_PATTERN, transform=pil_to_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)}")

    model = RetinaNet(config.NUM_CLASSES, get_backbone()).to(device)
    loss_fn = RetinaNetLoss()
    label_encoder = LabelEncoder()

    checkpoint_pattern = config.EPOCH_CHECKPOINT_TEMPLATE.format(epoch='*')
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(re.search(r'checkpoint_epoch_(\d+).pth', x).group(1)))

    if not checkpoint_paths:
        print("No epoch checkpoints found to evaluate.")
        return

    print("\n--- Evaluating All Epoch Checkpoints ---")
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', -1)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_images, gt_boxes, cls_ids in test_loader:
                encoded_images, encoded_labels = label_encoder.encode_batch(batch_images, gt_boxes, cls_ids)
                encoded_images = encoded_images.to(device)
                cls_targets, box_targets = encoded_labels[0].to(device), encoded_labels[1].to(device)
                
                predictions = model(encoded_images)
                loss = torch.mean(loss_fn(predictions, (cls_targets, box_targets)))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch {epoch + 1}: Test Loss = {avg_test_loss:.4f}")
    print("--------------------------------------")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "evaluate"]:
        print("Usage: python training_and_evaluation.py [train|evaluate]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "train":
        run_training()
    elif mode == "evaluate":
        run_evaluation()