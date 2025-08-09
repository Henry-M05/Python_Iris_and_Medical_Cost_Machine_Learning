# 7/11/2025

import kagglehub
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from torch import nn
import pandas as pd
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# from torchvision import datasets, transforms

# Download latest version
path = kagglehub.dataset_download("uciml/iris")
# https://www.kaggle.com/datasets/uciml/iris
path = path + "\\Iris.csv"
device = "cpu"

#print("Path to dataset files:", path)

classes = ["Species"]
continuous = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
target = "Species"



data = pd.read_csv(path)
data = data.values
print("Begin! -------------------------- Begin!")
# print(data)
splitData = torch.utils.data.random_split(data, [.7,.2,.1])
#print(splitData[0][:]) #First index chooses the seventy, twenty, or ten percent; second index shows data from that grouping

formattedData = pd.DataFrame(splitData[0][:])

formattedData.columns = ["Id", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
validationData = pd.DataFrame(splitData[1][:])
validationData.columns = ["Id", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
print(formattedData.columns)

print('''


wheeee!



''')

data_config = DataConfig( # Says which columns are the datums, which column is the target, and what the classes are.
    target=[target],
    continuous_cols=continuous,
    categorical_cols=classes,
    num_workers=11,
    pin_memory=False,
)

trainer_config = TrainerConfig(
    batch_size=10,
    max_epochs=70,
    accelerator=device,  # or "gpu" if available
    load_best=False,

)

model_config = CategoryEmbeddingModelConfig(
    layers="4-15-3",  # Example: two layers with 128 and 64 neurons
    activation="ReLU",
    dropout=0.1,
    task="classification",
    loss="CrossEntropyLoss", # automatically uses the CrossEntropyLoss function when task is classification regardless. This is written in for ✨learning✨
    metrics=["accuracy"], #same as above
    metrics_prob_input=[False],
    learning_rate=.001
    )


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(4,9001), #input, number of nodes. Each input connects to each and every node.
#             nn.ReLU(), # The NONLINEAR component (The Activation Function)
#             nn.Linear(9001, 15),
#             nn.ReLU(),
#             nn.Linear(15, 3),
#         )
#
#     def forward(self, x): #Forward is not called directly.
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x) # Runs the data through the model; saves output (which are the logits)
#         return logits # The raw output numbers from the final nodes.



# modelConfig = NeuralNetwork().to(device)
# modelConfig.train(True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



optimizer_config = OptimizerConfig(
    optimizer="SGD",
    optimizer_params = {
        "weight_decay": .00001,
        "momentum": 0,
        "dampening": 0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "differentiable": False,
        "fused": None
    }
)

model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

sModel = model.fit(
    train=formattedData, # Pass your training DataFrame
    validation=validationData,
    )

predictions = model.predict(validationData)
combo = pd.concat([predictions,validationData], axis=1)
print(combo.values)

# print(f"Model structure: {model}\n\n") # prints the structure of the neural network. Linear, ReLU, Linear, ReLU, Linear

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

