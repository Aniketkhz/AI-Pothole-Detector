from sklearn.metrics import precision_score, recall_score

def validate(model, dataloader, conf_thres=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for _, imgs, targets in dataloader:
            preds = model(imgs)
            preds = non_max_suppression(preds, conf_thres=conf_thres)
            
            # Process predictions and targets for metrics
            # ... (implementation depends on your exact format)
            
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')
    return precision, recall