import argparse
import torch
import torch.nn as nn
from utils import VGG, data_loader, train, evaluate, make_layers, cfgs_vgg

def main():
    parser = argparse.ArgumentParser(description='VGG')
    parser.add_argument('--VGG_version', type=int, default=16, choices=[11, 13, 16, 19])
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--data_directory', type=str, default="dataset/")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_classifier = VGG(make_layers(cfgs_vgg[args.VGG_version]), args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg_classifier.parameters(), lr= args.learning_rate)
    train_loader, valid_loader = data_loader(data_dir="dataset/", batch_size=args.batch_size)
    test_loader = data_loader(data_dir="dataset/", batch_size=args.batch_size, test=True)

    train(args.num_epochs, train_loader, valid_loader, vgg_classifier, criterion, optimizer, device)
    evaluate(test_loader, vgg_classifier, criterion, device)

if __name__ == '__main__':
    main()



