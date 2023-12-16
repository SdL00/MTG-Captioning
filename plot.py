import json
import matplotlib.pyplot as plt

def visualize_history(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Extract metrics and loss from history
    plt.figure(figsize=(20,8))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()

    plt.figure(figsize=(20,8))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Training and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()

# Visualize history of the last training
history_file_path = 'results_training/history.json'
visualize_history(history_file_path)
