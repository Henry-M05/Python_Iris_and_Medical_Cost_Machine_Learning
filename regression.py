# 8/2/2025

import kagglehub
import numpy
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from torch import nn
import pandas as pd
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datetime import datetime
# from torchvision import datasets, transforms

pd.set_option("display.max_rows", None)
# Load data
path = kagglehub.dataset_download("mirichoi0218/insurance") + "\\insurance.csv"
# https://www.kaggle.com/datasets/mirichoi0218/insurance
df = pd.read_csv(path)

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)

# Define columns
target = "charges"
continuous = ["age", "bmi", "children"]
categorical = ["sex", "smoker", "region"]

# Configurations
data_config = DataConfig(
    target=[target],
    continuous_cols=continuous,
    categorical_cols=categorical,
)

trainer_config = TrainerConfig(
    max_epochs=100,
    batch_size=32,
    load_best=False,
)

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers="64-32-16",
    activation="ReLU",
    learning_rate=1e-3,
)

optimizer_config = OptimizerConfig(optimizer="Adam")

# Create & train model
model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=trainer_config,
    optimizer_config=optimizer_config,
)

model.fit(train=train_df, validation=val_df)
preds = model.predict(val_df)
print(preds.describe())
#print(preds)

# Ensure preds is a DataFrame with column "prediction"
if not isinstance(preds, pd.DataFrame):
    preds = pd.DataFrame(preds, columns=["charges_prediction"])

# Reset indices
preds = preds.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Combine actual and predicted
comparison = pd.concat([val_df[[target]], preds], axis=1)

# Calculate errors
comparison["error"] = comparison["charges_prediction"] - comparison[target]
comparison["abs_error"] = comparison["error"].abs()

# Display
print(comparison#.head(10)
      )
print("\nError Summary:\n", comparison[["error", "abs_error"]].describe())
