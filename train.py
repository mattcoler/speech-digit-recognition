import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from model import DigitRecognizer

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (specs, labels) in enumerate(train_loader):
        specs, labels = specs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    average_loss = val_loss / len(val_loader)
    
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def plot_training_history(train_acc, val_acc, train_loss, val_loss):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train a speech digit recognizer')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='./models', help='directory to save models')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, valid_loader, digit_classes = get_dataloaders(batch_size=args.batch_size)
    
    # Print dataset information
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(valid_loader.dataset)}")
    
    # Create the model
    model = DigitRecognizer(num_classes=len(digit_classes))
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move the model to the device
    model = model.to(device)
    
    # Train the model
    best_accuracy = 0.0
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train for one epoch
        train_acc = train(model, train_loader, criterion, optimizer, device)
        train_accuracies.append(train_acc)
        
        # Evaluate on validation set
        val_acc = evaluate(model, valid_loader, criterion, device)
        val_accuracies.append(val_acc)
        
        # Save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_path = os.path.join(args.save_dir, 'digit_recognizer_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with accuracy: {best_accuracy:.2f}%')
        
        print('----------------------------')
    
    print(f'Training complete! Best accuracy: {best_accuracy:.2f}%')
    
    # Plot training history
    plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses)

if __name__ == '__main__':
    main()
