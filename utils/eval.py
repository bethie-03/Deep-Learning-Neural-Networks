import torch

def inference(test_loader, classifier, device):
    with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = classifier(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct +=(predicted == labels).sum().item()
                del images, labels, outputs

            print(f'Accuracy of the model on test data: {(correct/total)*100}%')