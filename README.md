# **Image Classification with Custom Dropout Implementation**

---

## **Project Description**

This project involves building an image classification model using a custom implementation of the dropout regularization technique. The primary objective is to create a neural network capable of classifying natural scene images into six categories while implementing dropout from scratch to induce sparsity and prevent overfitting.

### **Key Features:**
- **Custom Dropout Layer:** Implemented manually to provide a deeper understanding of dropout mechanics.
- **Sparsity Achievement:** The custom dropout layer encourages sparsity in the network, contributing to better generalization.
- **Image Classification:** The model classifies images into the following categories: *Buildings, Forest, Glacier, Mountain, Sea, Street*.
- **Memory Efficiency:** The dataset is preprocessed to avoid memory overload by resizing images and using a reduced batch size.
- **Visualization:** Plots training and validation loss and accuracy over epochs.

---

## **Dataset**

The dataset used for this project is the **Natural Scenes Dataset**, available on Kaggle under the title **"Natural Scene Classification"**. The dataset includes 25,000 images of 150x150 pixels across six categories:

- `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

The dataset is **not included** in this repository. Please download it manually from Kaggle:

- [Natural Scene Classification on Kaggle](https://www.kaggle.com/datasets) *(Provide the exact link if available)*

**After downloading:**
- Extract the dataset into a `data` directory with the following structure:

```plaintext
Dependencies
The project requires Python 3.8+ and the following libraries:

text
Copy
Edit
numpy
tensorflow
opencv-python
scikit-learn
matplotlib
pillow
Alternatively, install all dependencies using:

sh
Copy
Edit
pip install -r requirements.txt
requirements.txt:

text
Copy
Edit
numpy==1.23.5
tensorflow==2.13.0
opencv-python==4.8.0
scikit-learn==1.3.0
matplotlib==3.7.2
pillow==10.0.1
Setup and Usage
1. Clone the Repository
sh
Copy
Edit
git clone <repository-url>
cd <repository-directory>
2. Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
