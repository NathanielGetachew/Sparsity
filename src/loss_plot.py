import matplotlib.pyplot as plt

def save_loss_plot(history):
    # Extract loss and accuracy history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('training_loss.png')  # Save to file instead of plt.show()
    plt.close()  # Close the plot to free up memory
    
    # Plot training & validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('training_accuracy.png')  # Save to file instead of plt.show()
    plt.close()  # Close the plot to free up memory
