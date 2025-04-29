Skin Lesion Classification Using Deep Learning: A Comparative Study of ResNet50 and CNN

This project presents a deep learning-based approach to classify skin lesions as benign or malignant, comparing the performance of a custom Convolutional Neural Network (CNN) and a pre-trained ResNet50 model. The aim is to support early detection of skin cancer using image-based analysis.

Steps to Execute

1. Upload Dataset

Download the HAM10000 dataset from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Upload the entire dataset folder (including image files and metadata CSV) to your Google Drive.

2. Train Custom CNN Model

Open the notebook Skin_Lesion_CNN.ipynb in Google Colab

Mount your Google Drive in the notebook.

Ensure the dataset path in the notebook matches your Drive path.

Run all cells to preprocess the data, build the CNN model, train, and evaluate it.

3. Train ResNet50 Model

Open the notebook Skin_Lesion_ResNet50.ipynb in Google Colab

Mount your Google Drive in the notebook.

Verify the dataset path matches where your HAM10000 data is located.

Run all cells to preprocess data, load the pre-trained ResNet50, fine-tune it, and evaluate the model.

4. Compare Results

Review the accuracy, loss graphs, and classification reports from both notebooks.

Compare the performance of the custom CNN and ResNet50 models in terms of accuracy, precision, recall, and F1-score.

5. Access Model Outputs

Trained model files, graphs, and evaluation metrics will be saved to your Google Drive.

You can download the saved models or use them for deployment or further fine-tuning
