import torch
import json
import matplotlib.pyplot as plt

def visualize_logs():
    with open('history/train_logs.json', 'r') as f:
        train_logs = json.load(f)
    
    with open('history/validation_logs.json', 'r') as f:
        validate_logs = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(train_logs['Loss'], label='Train Loss')
    plt.plot(validate_logs['Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig('history/loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_logs['Accuracy'], label='Train Accuracy')
    plt.plot(validate_logs['Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig('history/accuracy.png')
    plt.close()
    
def train(num_epochs, train_loader, valid_loader, classifier, metrics, criterion, optimizer, device):
    best_val_loss = float('inf')
    history_train_logs = {'Loss': []}
    history_val_logs = {'Loss': []}
    
    for epoch in range(num_epochs):
        #training
        classifier.train()
        total_train_loss = 0.0
        for metric in metrics.values():
            metric.reset()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total_train_loss += loss.item()
            for metric in metrics.values():
                metric.update(predicted, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / len(train_loader)
        history_train_logs['Loss'].append(average_train_loss)
                
        result = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}"
        
        for metric_name, metric in zip(metrics.keys(), metrics.values()):
            metric_value = metric.compute().item()
            result += f', {metric_name}: {metric_value:.2f}'
            if epoch == 0:
                history_train_logs[f'{metric_name}'] = [metric_value]
            else:
                history_train_logs[f'{metric_name}'].append(metric_value)
            
        #validating
        with torch.no_grad():
            classifier.eval()
            total_val_loss = 0.0
            for metric in metrics.values():
                metric.reset()
            
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = classifier(images)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                total_val_loss += val_loss.item()
                for metric in metrics.values():
                    metric.update(predicted, labels)

                del images, labels, outputs

            average_val_loss = total_val_loss / len(valid_loader)
            history_val_logs['Loss'].append(average_val_loss)
            
            result += f', Val Loss: {average_val_loss:.4f}'
            for metric_name, metric in zip(metrics.keys(), metrics.values()):
                metric_value = metric.compute().item()
                result += f', Val {metric_name}: {metric_value:.2f}'
                if epoch == 0:
                    history_val_logs[f'{metric_name}'] = [metric_value]
                else:
                    history_val_logs[f'{metric_name}'].append(metric_value)
        
        print(result)
        if average_val_loss < best_val_loss:
            torch.save(classifier.state_dict(), 'weights/best.pt')
            best_val_loss = average_val_loss

    torch.save(classifier.state_dict(), 'weights/last.pt')

    with open('history/train_logs.json', 'w') as f:
        json.dump(history_train_logs, f, indent=4)
        
    with open('history/validation_logs.json', 'w') as f:
        json.dump(history_val_logs, f, indent=4)

    visualize_logs()
