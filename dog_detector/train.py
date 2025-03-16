#dot_detector/train.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
from dog_detector.config import config
from dog_detector.utils import assign_anchors_to_image, compute_iou
from dog_detector.visualization import visualize_predictions

def train_one_epoch(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = len(dataloader)
    cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    reg_loss_fn = nn.SmoothL1Loss(reduction="sum")
    
    progress_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch} Training")
    for batch_idx, (images, targets) in progress_bar:
        images = images.to(device)
        cls_out, reg_out = model(images)
        batch_size, _, feat_h, feat_w = cls_out.shape
        
        # Generate anchors for this feature map size
        anchors = model.generate_anchors((feat_h, feat_w), device)
        
        # Reshape outputs properly based on feature map dimensions
        # First reshape: [batch, channels, height, width] -> [batch, 2, num_anchors, height, width]
        cls_out = cls_out.reshape(batch_size, 2, model.num_anchors, feat_h, feat_w)
        # Then permute to [batch, height, width, num_anchors, 2]
        cls_out = cls_out.permute(0, 3, 4, 2, 1)
        # Finally reshape to [batch*height*width*num_anchors, 2]
        cls_out_flat = cls_out.reshape(-1, 2)
        
        # Similar for regression outputs
        reg_out = reg_out.reshape(batch_size, 4, model.num_anchors, feat_h, feat_w)
        reg_out = reg_out.permute(0, 3, 4, 2, 1)
        reg_out_flat = reg_out.reshape(-1, 4)
        
        optimizer.zero_grad()
        
        batch_cls_labels = []
        batch_reg_targets = []
        batch_reg_masks = []
        
        # Process each image in the batch
        for i, target in enumerate(targets):
            # Extract boxes tensor from target dictionary and ensure it's on the right device
            gt_boxes = target["boxes"].to(device)
            # Scale ground truth boxes to match the image size used by the model
            if gt_boxes.numel() > 0:
                img_height, img_width = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
                orig_height, orig_width = target.get("orig_size", (img_height, img_width))
                
                if orig_height != img_height or orig_width != img_width:
                    # Scale the boxes
                    scale_x = img_width / orig_width
                    scale_y = img_height / orig_height
                    
                    gt_boxes[:, 0] *= scale_x  # x1
                    gt_boxes[:, 1] *= scale_y  # y1
                    gt_boxes[:, 2] *= scale_x  # x2
                    gt_boxes[:, 3] *= scale_y  # y2
            
            # Assign anchors to ground truth boxes
            cls_labels, reg_targets, reg_mask = assign_anchors_to_image(anchors, gt_boxes)
            batch_cls_labels.append(cls_labels)
            batch_reg_targets.append(reg_targets)
            batch_reg_masks.append(reg_mask)
        
        batch_cls_labels = torch.cat(batch_cls_labels).to(device)
        batch_reg_targets = torch.cat(batch_reg_targets).to(device)
        batch_reg_masks = torch.cat(batch_reg_masks).to(device)
        
        # Calculate classification loss
        cls_loss = cls_loss_fn(cls_out_flat, batch_cls_labels) / (batch_cls_labels.size(0) or 1)
        
        # Only compute regression loss if there are positive anchors
        if batch_reg_masks.any():
            reg_loss = reg_loss_fn(reg_out_flat[batch_reg_masks], batch_reg_targets[batch_reg_masks]) / (batch_reg_masks.sum().item() or 1)
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # Calculate total loss with weighted regression component
        loss = cls_loss + config.REG_LOSS_WEIGHT * reg_loss
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_reg_loss += reg_loss.item() * images.size(0)
        
        # Log per-step metrics
        global_step = (epoch - 1) * num_batches + batch_idx
        writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
        writer.add_scalar("Train/BatchClsLoss", cls_loss.item(), global_step)
        writer.add_scalar("Train/BatchRegLoss", reg_loss.item(), global_step)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls_loss': f"{cls_loss.item():.4f}",
            'reg_loss': f"{reg_loss.item():.4f}"
        })
    
    progress_bar.close()
    
    # Calculate average losses per sample
    dataset_size = len(dataloader.dataset)
    avg_loss = total_loss / dataset_size
    avg_cls_loss = total_cls_loss / dataset_size
    avg_reg_loss = total_reg_loss / dataset_size
    
    metrics = {
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "reg_loss": avg_reg_loss
    }
    
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f})")
    
    # Add epoch metrics to tensorboard
    writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
    writer.add_scalar("Train/EpochClsLoss", avg_cls_loss, epoch)
    writer.add_scalar("Train/EpochRegLoss", avg_reg_loss, epoch)
    
    return metrics

def evaluate(model, dataloader, device, epoch, writer, log_images=False):
    model.eval()
    pred_counts = []
    confs = []
    ious = []
    true_positives = 0
    false_positives = 0
    total_gt_boxes = 0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    reg_loss_fn = nn.SmoothL1Loss(reduction="sum")
    
    # For image logging
    images_logged = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} Evaluation")
    with torch.no_grad():
        for batch_idx, (images, targets) in progress_bar:
            images = images.to(device)
            cls_out, reg_out = model(images)
            batch_size, _, feat_h, feat_w = cls_out.shape
            anchors = model.generate_anchors((feat_h, feat_w), device)
            
            # Process outputs using the same reshaping as in training
            cls_out = cls_out.reshape(batch_size, 2, model.num_anchors, feat_h, feat_w)
            cls_out = cls_out.permute(0, 3, 4, 2, 1).contiguous()
            cls_out_flat = cls_out.reshape(-1, 2)
            
            reg_out = reg_out.reshape(batch_size, 4, model.num_anchors, feat_h, feat_w)
            reg_out = reg_out.permute(0, 3, 4, 2, 1).contiguous()
            reg_out_flat = reg_out.reshape(-1, 4)
            
            # Calculate losses for evaluation
            batch_cls_labels = []
            batch_reg_targets = []
            batch_reg_masks = []
            
            for i, target in enumerate(targets):
                gt_boxes = target["boxes"].to(device)
                # Scale ground truth boxes if needed (same as in training)
                if gt_boxes.numel() > 0:
                    img_height, img_width = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
                    orig_height, orig_width = target.get("orig_size", (img_height, img_width))
                    
                    if orig_height != img_height or orig_width != img_width:
                        scale_x = img_width / orig_width
                        scale_y = img_height / orig_height
                        
                        gt_boxes[:, 0] *= scale_x  # x1
                        gt_boxes[:, 1] *= scale_y  # y1
                        gt_boxes[:, 2] *= scale_x  # x2
                        gt_boxes[:, 3] *= scale_y  # y2
                
                cls_labels, reg_targets, reg_mask = assign_anchors_to_image(anchors, gt_boxes)
                batch_cls_labels.append(cls_labels)
                batch_reg_targets.append(reg_targets)
                batch_reg_masks.append(reg_mask)
            
            batch_cls_labels = torch.cat(batch_cls_labels).to(device)
            batch_reg_targets = torch.cat(batch_reg_targets).to(device)
            batch_reg_masks = torch.cat(batch_reg_masks).to(device)
            
            # Calculate losses - normalize by batch size for consistency
            cls_loss = cls_loss_fn(cls_out_flat, batch_cls_labels) / (batch_cls_labels.size(0) or 1)
            if batch_reg_masks.any():
                reg_loss = reg_loss_fn(reg_out_flat[batch_reg_masks], batch_reg_targets[batch_reg_masks]) / (batch_reg_masks.sum().item() or 1)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            total_cls_loss += cls_loss.item() * images.size(0)
            total_reg_loss += reg_loss.item() * images.size(0)
            
            # Get predictions using post_process method - pass the original outputs from model
            # Not the reshaped outputs we use for loss calculation
            orig_cls_out, orig_reg_out = model(images)
            boxes_list, scores_list = model.post_process(orig_cls_out, orig_reg_out, anchors)
            
            for i in range(batch_size):
                if log_images and images_logged < config.NUM_VAL_IMAGES_TO_LOG:
                    fig = visualize_predictions(images[i], targets[i], boxes_list[i], scores_list[i])
                    writer.add_figure(f"Validation/Image_{images_logged+1}", fig, epoch)
                    plt.close(fig)
                    images_logged += 1
                
                # Continue with normal evaluation metrics
                pred_boxes = boxes_list[i]
                pred_scores = scores_list[i]
                count = pred_boxes.shape[0]
                pred_counts.append(count)
                
                if count > 0:
                    confs.extend(pred_scores.cpu().numpy())
                
                gt_boxes = targets[i]["boxes"].to(device)
                if count > 0 and gt_boxes.numel() > 0:
                    # Scale ground truth boxes if needed (as above)
                    img_height, img_width = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]
                    orig_height, orig_width = targets[i].get("orig_size", (img_height, img_width))
                    
                    if orig_height != img_height or orig_width != img_width:
                        scale_x = img_width / orig_width
                        scale_y = img_height / orig_height
                        
                        gt_boxes[:, 0] *= scale_x
                        gt_boxes[:, 1] *= scale_y
                        gt_boxes[:, 2] *= scale_x
                        gt_boxes[:, 3] *= scale_y
                    
                    iou_matrix = compute_iou(pred_boxes, gt_boxes)
                    max_ious, _ = iou_matrix.max(dim=1)
                    avg_iou = max_ious.mean().item() if len(max_ious) > 0 else 0.0
                else:
                    avg_iou = 0.0
                ious.append(avg_iou)

                # Calculate TP/FP
                total_gt_boxes += len(gt_boxes)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou_matrix = compute_iou(pred_boxes, gt_boxes)
                    for pred_idx in range(len(pred_boxes)):
                        if pred_scores[pred_idx] > config.CONF_THRESHOLD:
                            max_iou, _ = iou_matrix[pred_idx].max(dim=0)
                            if max_iou >= config.IOU_THRESHOLD:
                                true_positives += 1
                            else:
                                false_positives += 1
                elif len(pred_boxes) > 0:
                    # All predictions are false positives if there are no ground truth boxes
                    false_positives += len([s for s in pred_scores if s > config.CONF_THRESHOLD])

    # Calculate metrics
    mean_pred_count = sum(pred_counts) / len(pred_counts) if pred_counts else 0
    mean_conf = sum(confs) / len(confs) if confs else 0
    mean_iou = sum(ious) / len(ious) if ious else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average losses
    dataset_size = len(dataloader.dataset)
    avg_cls_loss = total_cls_loss / dataset_size
    avg_reg_loss = total_reg_loss / dataset_size
    total_loss = avg_cls_loss + config.REG_LOSS_WEIGHT * avg_reg_loss
    
    # Add loss metrics to tensorboard
    writer.add_scalar("Eval/Loss", total_loss, epoch)
    writer.add_scalar("Eval/ClsLoss", avg_cls_loss, epoch)
    writer.add_scalar("Eval/RegLoss", avg_reg_loss, epoch)
    
    # Log scalar metrics
    writer.add_scalar("Eval/MeanPredCount", mean_pred_count, epoch)
    writer.add_scalar("Eval/MeanConfidence", mean_conf, epoch)
    writer.add_scalar("Eval/MeanIoU", mean_iou, epoch)
    writer.add_scalar("Eval/Precision", precision, epoch)
    writer.add_scalar("Eval/Recall", recall, epoch)
    writer.add_scalar("Eval/F1Score", f1_score, epoch)
    
    metrics = {
        "mean_pred_count": mean_pred_count,
        "mean_confidence": mean_conf,
        "mean_iou": mean_iou,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "total_gt_boxes": total_gt_boxes,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "loss": total_loss,
        "cls_loss": avg_cls_loss,
        "reg_loss": avg_reg_loss
    }
    
    print(f"Evaluation at epoch {epoch} complete:")
    print(f"  Mean IoU: {mean_iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"  Loss: {total_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f})")
    
    return metrics
