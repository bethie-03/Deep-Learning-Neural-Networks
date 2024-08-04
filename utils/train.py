import torch
import json
import matplotlib.pyplot as plt

def visualize_logs():
    with open('history/training_logs.json', 'r') as f:
        logs = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(logs['train_loss'], label='Train Loss')
    plt.plot(logs['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig('history/loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(logs['train_accuracy'], label='Train Accuracy')
    plt.plot(logs['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig('history/accuracy.png')
    plt.close()
    
def train(num_epochs, train_loader, valid_loader, classifier, criterion, optimizer, device):
    best_val_loss = float('inf')

    history_logs = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        classifier.train()
        correct_train = 0
        total_train = 0
        total_train_loss = 0.0  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            total_train_loss += loss.item()  

            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100

        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_val_loss = 0.0

            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = classifier(images)
                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_val_loss += val_loss.item()

                del images, labels, outputs

            accuracy = (correct / total) * 100
            average_val_loss = total_val_loss / len(valid_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {average_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

        history_logs['train_loss'].append(average_train_loss)
        history_logs['train_accuracy'].append(train_accuracy)
        history_logs['val_loss'].append(average_val_loss)
        history_logs['val_accuracy'].append(accuracy)

        if average_val_loss < best_val_loss:
            torch.save(classifier.state_dict(), 'weights/best.pt')
            best_val_loss = average_val_loss

    torch.save(classifier.state_dict(), 'weights/last.pt')

    with open('history/training_logs.json', 'w') as f:
        json.dump(history_logs, f, indent=4)

    visualize_logs()

if __name__ == '__main__':
    visualize_logs()