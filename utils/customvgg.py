import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features, num_classes, init_weight = True):
        super(VGG, self).__init__()

        self.features_layer = features
        self.dense_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, num_classes)
        )

        if init_weight:
            self.initialize_weight()
        
    def forward(self, x):
        out = self.features_layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.dense_layer(out)
        return out
    
    def initialize_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, 0)

def make_layers(cfg, batch_norm = True):
    layers = []
    in_channels = 3

    for v in cfg:
        if v=='M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(v)]
            layers+= [nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)



