import torch
import torch.nn.functional as F
import sys
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# Add the parent directory to the path to import the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training_and_evaluation import (
    RetinaNetLoss,
    RetinaNetClassificationLoss,
    RetinaNetBoxLoss,
    LabelEncoder,
    TrainingConfig,
    TFRecordKITTIDataset,
    collate_fn
)

def test_loss_with_real_data():
    """
    Tests the RetinaNet loss components with a real data sample from TFRecord.
    """
    print("\n--- Running Loss Calculation Test with Real Data ---")

    # 1. Configuration
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 2
    NUM_CLASSES = config.NUM_CLASSES

    # 2. Load a Real Data Batch
    pil_to_tensor = transforms.Compose([transforms.ToTensor()])
    # Using validation set for testing is a good practice
    dataset = TFRecordKITTIDataset(config.VALIDATION_TFRECORD_PATTERN, transform=pil_to_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    try:
        real_images, real_gt_boxes, real_cls_ids = next(iter(loader))
    except StopIteration:
        print("Could not load data. Make sure TFRecord files are available at:", config.VALIDATION_TFRECORD_PATTERN)
        return

    print(f"\nLoaded a real data batch of size {BATCH_SIZE}.")
    for i in range(BATCH_SIZE):
        print(f"Sample {i+1} has {len(real_gt_boxes[i])} objects.")

    # 3. Encode Ground Truth Labels
    label_encoder = LabelEncoder()
    _, (cls_targets, box_targets) = label_encoder.encode_batch(real_images, real_gt_boxes, real_cls_ids)
    
    cls_targets = cls_targets.to(device)
    box_targets = box_targets.to(device)

    num_anchors = cls_targets.shape[1]
    print(f"\nEncoded ground truth labels. Number of anchors per image: {num_anchors}")
    print(f"Shape of cls_targets: {cls_targets.shape}")
    print(f"Shape of box_targets: {box_targets.shape}")

    # 4. Create Mock Model Predictions
    pred_cls = torch.randn(BATCH_SIZE, num_anchors, NUM_CLASSES, device=device)
    pred_box = torch.randn(BATCH_SIZE, num_anchors, 4, device=device)
    
    mock_predictions = (pred_cls, pred_box)
    print("\nCreated mock model predictions.")
    print(f"Shape of pred_cls: {pred_cls.shape}")
    print(f"Shape of pred_box: {pred_box.shape}")

    # 5. Calculate Loss
    clf_loss_fn = RetinaNetClassificationLoss(config.CLASSIFICATION_LOSS_ALPHA, config.CLASSIFICATION_LOSS_GAMMA)
    box_loss_fn = RetinaNetBoxLoss(config.BOX_LOSS_DELTA)
    total_loss_fn = RetinaNetLoss()

    # --- Detailed Loss Calculation ---
    positive_mask = (cls_targets > -1.0).float()
    ignore_mask = (cls_targets == -2.0).float()
    normalizer = torch.sum(positive_mask, dim=-1).clamp(min=1.0)

    # ADDED: Print the number of positive anchors
    num_positive_anchors = torch.sum(positive_mask, dim=-1)
    print(f"\nNumber of positive anchors found (per sample): {num_positive_anchors.cpu().detach().numpy()}")

    # Classification Loss
    temp_cls_targets = torch.where(cls_targets >= 0, cls_targets.long(), 0)
    one_hot_labels = F.one_hot(temp_cls_targets, num_classes=NUM_CLASSES).float()
    one_hot_labels = one_hot_labels * positive_mask.unsqueeze(-1)
    
    clf_loss = clf_loss_fn(one_hot_labels, pred_cls.float())
    clf_loss = torch.where(ignore_mask == 1.0, 0.0, clf_loss)
    clf_loss_per_sample = torch.sum(clf_loss, dim=-1) / normalizer

    # Box Loss
    box_loss = box_loss_fn(box_targets.float(), pred_box.float())
    box_loss = torch.where(positive_mask == 1.0, box_loss, 0.0)
    box_loss_per_sample = torch.sum(box_loss, dim=-1) / normalizer

    # Total Loss (using the combined class)
    total_loss = total_loss_fn(mock_predictions, (cls_targets, box_targets))

    print("\n--- Loss Calculation Results ---")
    print(f"Classification Loss (per sample): {clf_loss_per_sample.cpu().detach().numpy()}")
    print(f"Box Loss (per sample):          {box_loss_per_sample.cpu().detach().numpy()}")
    print(f"Combined Loss (per sample):     {(clf_loss_per_sample + box_loss_per_sample).cpu().detach().numpy()}")
    print(f"Total Loss from RetinaNetLoss class: {total_loss.cpu().detach().numpy()}")
    
    # 6. Verification
    assert total_loss.shape == (BATCH_SIZE,), f"Expected loss shape ({BATCH_SIZE},), but got {total_loss.shape}"
    mean_loss = torch.mean(total_loss)
    assert mean_loss.ndim == 0, f"Expected mean loss to be a scalar, but got {mean_loss.ndim} dimensions"
    
    print("\nTest Passed: Loss calculation is behaving as expected with real data.")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    test_loss_with_real_data()
