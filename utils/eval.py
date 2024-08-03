import torch

def evaluate(test_loader, classifier, criterion, device):
    classifier.eval()  
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0.0  

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier(images)
            loss = criterion(outputs, labels)  

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()  

        accuracy = (correct / total) * 100
        average_loss = total_loss / len(test_loader)  

        print(f'Accuracy of the model on test data: {accuracy:.2f}%')
        print(f'Average loss of the model on test data: {average_loss:.4f}')