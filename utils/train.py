import torch

def train(num_epochs, train_loader, valid_loader, classifier, criterion, optimizer, device):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            #Forward Pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            #Back Propagation and optimizers
            optimizer.zero_grad()  #W_new = W_old - LR * Gradient
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1, num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = classifier(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct +=(predicted == labels).sum().item()
                del images, labels, outputs

            print(f'Accuracy of the model on validation data: {(correct/total)*100}%')







