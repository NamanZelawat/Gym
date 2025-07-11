"""
Trains a PyTorch image classification mdel using device-agnostic code.
"""
import os
import torch

from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train/"
test_dir = "data/pizza_steak_sushi/test/"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create the model
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                            lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device
)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file
utils.save_model(
    model=model,
    target_dir="models",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth"
)
