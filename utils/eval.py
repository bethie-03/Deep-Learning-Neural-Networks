import torch

def evaluate(test_loader, classifier, metrics, criterion, device):
    classifier.eval()

    with torch.no_grad():
        total_loss = 0.0
    
        for metric in metrics.values():
                metric.reset()

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item()
            
            for metric in metrics.values():
                metric.update(predicted, labels)

        average_loss = total_loss / len(test_loader)
        
        result = f"Loss: {average_loss:.4f}"
        
        for metric_name, metric in zip(metrics.keys(), metrics.values()):
            result += f', {metric_name}: {metric.compute().item():.2f}'
        print(result)