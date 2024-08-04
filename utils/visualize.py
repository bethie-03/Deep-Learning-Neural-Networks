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

    plt.figure(figsize=(10, 5))
    plt.plot(logs['train_accuracy'], label='Train Accuracy')
    plt.plot(logs['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig('history/accuracy.png')