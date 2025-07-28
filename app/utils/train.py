import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels
from utils.utils import non_max_suppression, plot_images
from models import Darknet
import time
import os
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=416, help='image size')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/pothole.data', help='data config file')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    opt = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Darknet('config/yolov3.cfg').to(device)
    model.load_darknet_weights(opt.weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Loss function
    criterion = nn.MSELoss(reduction='none')  # Using MSE for coordinates and confidence

    # Dataset with enhanced augmentation
    train_dataset = LoadImagesAndLabels(
        'data/train.txt',
        img_size=opt.img_size,
        augment=True,
        augment_params={
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.1,
            'perspective': 0.001,
            'flipud': True,
            'fliplr': True,
            'mosaic': True,  # Enable mosaic augmentation
            'mixup': 0.2     # Enable mixup with 20% probability
        }
    )

    val_dataset = LoadImagesAndLabels('data/val.txt', img_size=opt.img_size, augment=False)

    # Dataloaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=opt.batch_size, 
                            shuffle=True,
                            num_workers=opt.workers,
                            pin_memory=True)

    val_loader = DataLoader(val_dataset, 
                          batch_size=opt.batch_size, 
                          shuffle=False,
                          num_workers=opt.workers,
                          pin_memory=True)

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        
        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(imgs)

            # Calculate loss
            loss = compute_loss(preds, targets, model, criterion)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            if batch_i % 50 == 0:
                print(f'Epoch [{epoch}/{opt.epochs}], Batch [{batch_i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validate after each epoch
        val_loss, val_precision, val_recall = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch}/{opt.epochs}] completed in {(time.time()-start_time):.2f}s')
        print(f'Validation - Loss: {val_loss:.4f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'checkpoints/epoch_{epoch}.pt')

def compute_loss(preds, targets, model, criterion):
    """Custom loss calculation for YOLOv3"""
    # Convert predictions to same format as targets
    # This is simplified - actual implementation would need to match your model's output format
    loss = criterion(preds, targets)
    return loss.mean()

def validate(model, dataloader, criterion, device, conf_thres=0.5, iou_thres=0.5):
    """Validation with metrics calculation"""
    model.eval()
    total_loss = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for _, imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(imgs)
            
            # Calculate loss
            loss = compute_loss(preds, targets, model, criterion)
            total_loss += loss.item()
            
            # Apply NMS and calculate metrics
            preds = non_max_suppression(preds, conf_thres, iou_thres)
            
            # Convert targets and predictions to comparable format
            # This part needs to match your specific data format
            # For demonstration, we'll assume simple comparison
            tp, fp, fn = calculate_metrics(preds, targets)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-16)
    recall = true_positives / (true_positives + false_negatives + 1e-16)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, precision, recall

def calculate_metrics(preds, targets):
    """Simplified metric calculation - adapt to your specific needs"""
    # This is a placeholder - implement according to your label format
    tp = 0
    fp = 0
    fn = 0
    
    # Compare predictions with ground truth
    # Actual implementation would need to match your data format
    return tp, fp, fn

if __name__ == '__main__':
    train()