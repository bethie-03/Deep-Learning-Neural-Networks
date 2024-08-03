# VGG16

I have completed training the VGG16 model from scratch based on the video provided on Notion. You can view the notebook file with the model's output results on the CIFAR100 dataset.

Additionally, I have expanded on this work by defining a CustomVGG class with configurations for different VGG versions and including weight initialization (see customvgg.py in the utils folder). During training, I implemented saving the best and last weights (see train.py in the utils folder), as well as calculating the loss on the validation and test sets.