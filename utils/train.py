import torch

def train(num_epochs, train_loader, valid_loader, classifier, criterion, optimizer, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  #W_new = W_old - LR * Gradient
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0.0

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = classifier(images)
                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct +=(predicted == labels).sum().item()
                total_loss += val_loss.item()

                del images, labels, outputs

            accuracy = (correct / total) * 100
            average_loss = total_loss / len(valid_loader)

        print(f"Epoch [{epoch+1, num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Val-loss: {average_loss:.4f}, Val-Accuracy: {accuracy:.2f}%")

        if average_loss < best_val_loss:
            torch.save(classifier.state_dict(), f'weights/best.pt')
            best_val_loss = average_loss

    torch.save(classifier.state_dict(), f'weights/last.pt')








