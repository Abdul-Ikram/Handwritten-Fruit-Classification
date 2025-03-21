# 🍎 Handwritten Urdu Fruit Name Classifier

This is a Django-based web application that classifies fruit images using a pre-trained deep learning model. Users can upload an image of a fruit, and the model predicts its category with a confidence score. The app also provides descriptions in English and Urdu for the predicted fruit.

## 🚀 Features

-   Upload fruit images for classification
-   Deep learning model for fruit recognition
-   Displays fruit details (name, description in English & Urdu)
-   Provides example images of the classified fruit
-   Handles errors for unclear images

## 🏗️ Model Architecture

The model used in this project is based on **Convolutional Neural Networks (CNNs)**, trained using **TensorFlow/Keras**. The architecture consists of:

-   **Convolutional Layers**: Extract spatial features from images
-   **Batch Normalization & Dropout**: Improve generalization and reduce overfitting
-   **Flatten & Fully Connected Layers**: Perform final classification
-   **Softmax Activation**: Assign probability scores for each fruit category

The model was trained using **Adam optimizer** with **Categorical Crossentropy Loss** and achieved **97% accuracy on training data and 99% on validation data**.

## 📂 Dataset

The model is trained on a **Handwritten Text Fruit Classification Dataset**, which contains labeled images of various fruits.

🔗 **Dataset Link**: [Kaggle - Handwritten Text Fruit Classification](https://www.kaggle.com/datasets/abdulikram/handwritten-text-fruit-classification)

## 🛠️ Technologies Used

-   **Django** – Backend framework
-   **TensorFlow/Keras** – Deep learning model for classification
-   **OpenCV** – Image processing
-   **NumPy** – Handling image arrays
-   **Joblib** – Loading model class labels

## 📂 Project Structure

```
Handwrite2Fruit/
  ├── .venv/
  ├── Handwrite2Fruit/
  │   ├── __pycache__/
  │   ├── __init__.py
  │   ├── asgi.py
  │   ├── settings.py
  │   ├── urls.py
  │   ├── wsgi.py
  │
  ├── userflow/
  │   ├── __pycache__/
  │   ├── migrations/
  │   ├── saved/
  │   │   ├── fruits_class_indices.joblib
  │   │   ├── fruits_classification_model.h5
  │   ├── static/
  │   │   ├── css/
  │   │   ├── images/
  │   │   ├── js/
  │   ├── templates/
  │   │   ├── index.html
  │   ├── __init__.py
  │   ├── admin.py
  │   ├── apps.py
  │   ├── models.py
  │   ├── tests.py
  │   ├── urls.py
  │   ├── views.py
  │
  ├── .gitignore
  ├── db.sqlite3
  ├── manage.py
  ├── requirements.txt


```

## 🖼️ How It Works

1.  User uploads an image of a fruit.
2.  The image is processed using OpenCV (grayscale, resized).
3.  The model predicts the fruit class and confidence score.
4.  If confidence is above 50%, fruit details are fetched from the database.
5.  The user sees the predicted fruit name, description, and example images.

## 🏗️ Setup & Installation

### Clone the Repository

```bash
git clone https://github.com/Abdul-Ikram/Handwritten-Fruit-Classification.git
cd Handwritten-Fruit-Classification

```

### Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # On Mac: source venv/bin/.activate

```

### Install Dependencies

```bash
pip install -r requirements.txt

```

### Run Migrations

```bash
python manage.py migrate

```

### Start the Django Server

```bash
python manage.py runserver

```

Now, open your browser and visit **`http://127.0.0.1:8000/api/classify/`** to use the app.

## 🔥 Notes

-   Ensure you have the saved model and class labels in `userflow/saved/`.
-   The model file should be named `fruits_classification_model (150_0.97_0.99).h5`.
-   The class labels should be in `fruits_class_indices.joblib`.
-   Update database records with fruit details before running the app.

## 📝 License

This project is open-source. Feel free to use and improve it!

----------
