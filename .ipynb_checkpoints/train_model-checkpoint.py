#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import argparse
import sys

#TODO: Import dependencies for Debugging andd Profiling
from smdebug import modes
import smdebug.pytorch as smd
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# this fix the 'OSError: image file is truncated (150 bytes not processed)' error 
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Testing
    print("START TESTING")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    corrects = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds==labels.data).item()
    accuracy = corrects / len(test_loader.dataset)
    loss = test_loss / len(test_loader.dataset)
    logger.info(f"Loss: {loss}, Accuracy: {100*accuracy}%")

def train(model, train_loader, validation_loader, criterion, optimizer, epoch, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for e in range(epoch):
        print(f"Epochs: {e}")
        # Training
        print("START TRAINING")
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        train_loss=0
        for inputs, labels in train_loader:
            optimizer.zero_grad() # Resets gradients for new batch
            outputs = model(inputs) # Runs Forwards Pass
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # Calculates Gradients for Model Parameters                
            optimizer.step() # Updates Weights
            train_loss += loss.item()
  
        # Validation
        print("START VALIDATING")
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        corrects = 0
        with torch.no_grad():
           for inputs, labels in validation_loader:
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              _, preds = torch.max(outputs, 1)
              corrects += torch.sum(preds == labels.data).item()
           accuracy = corrects / len(validation_loader.dataset)
           logger.info(f"Accuracy: {100*accuracy}%")
    
    return model 

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133)) # number of breeds
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # ToTensor moved at the bottom to solve this error: https://stackoverflow.com/questions/57079219/img-should-be-pil-image-got-class-torch-tensor
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),       # Data Augmentation
        transforms.Resize((224,224)),    # Resize image
        transforms.ToTensor()                        # Transforms image to range of 0 - 1
    ])

    testing_transform = transforms.Compose([          # No Data Augmentation for test transform
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(data, "train")
    validation_path = os.path.join(data, "valid")
    test_path = os.path.join(data, "test")
    
    
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False
    )
    
    validationset = torchvision.datasets.ImageFolder(root=validation_path, transform=testing_transform)
    validation_loader = torch.utils.data.DataLoader(
        validationset,
        batch_size=batch_size,
        shuffle=False
    )
    
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, validation_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    print(f"Running on Device {device}")
 
    model=net()
    model=model.to(device) # just to use the GPU
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()    
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader, test_loader = create_data_loaders(data=args.data, batch_size=args.batch_size)
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epoch, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, hook)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, "model"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args=parser.parse_args()
    
    main(args)
