import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from torchvision.models import resnet18


class CustomResNet18(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=4):
        super(CustomResNet18, self).__init__(observation_space, features_dim)

        # Load the pre-trained ResNet18 model
        resnet = resnet18(pretrained=True)
        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove the fully connected layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Define the custom linear layers
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(True),
            nn.Linear(256, features_dim),
        )

    def forward(self, observations):
        x = self.resnet(observations).squeeze(-1).squeeze(-1)
        return self.fc(x)


class CustomResNetPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomResNetPolicy, self).__init__(*args, **kwargs)
        kwargs["features_extractor_class"] = CustomResNet18
        kwargs["features_extractor_kwargs"] = dict(features_dim=4)
