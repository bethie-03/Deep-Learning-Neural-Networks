import argparse
from utils.vgg16 import *
from utils.train import *
from utils.eval import *
from utils.data_loader import *

def main():
    parser = argparse.ArgumentParser(description='VGG16')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_directory', type=str, default="dataset/")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16_classifier = VGG16(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg16_classifier.parameters(), lr= args.learning_rate)
    train_loader, valid_loader = data_loader(data_dir="dataset/", batch_size=args.batch_size)
    test_loader = data_loader(data_dir="dataset/", batch_size=args.batch_size, test=True)

    train(args.num_epochs, train_loader, valid_loader, vgg16_classifier, criterion, optimizer, device)
    eval(test_loader, vgg16_classifier, device)

if __name__ == '__main__':
    main()



