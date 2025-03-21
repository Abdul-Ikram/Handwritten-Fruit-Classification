from django.shortcuts import render
import cv2
import numpy as np
import tensorflow
import joblib
from .models import FruitsData

# Create your views here.

def classify(request):
    model = tensorflow.keras.models.load_model("userflow/saved/fruits_classification_model (150_0.97_0.99).h5")  # type: ignore
    class_labels = joblib.load("userflow/saved/fruits_class_indices.joblib")
    class_labels = {v: k for k, v in class_labels.items()}

    result = image_url_1 = image_url_2 = error_message = ''

    if request.method == 'POST':
        image_file = request.FILES.get('imageFile')

        if image_file:
            # Read the uploaded image using OpenCV
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize image
            resized_image = cv2.resize(gray_image, (128, 128))

            # Expand dimensions for model input
            image_array = np.expand_dims(resized_image, axis=0)

            # Generate the prediction
            prediction = model.predict(image_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(prediction)

            # Get the confidence score (probability) of the predicted class
            confidence_score = prediction[0][predicted_class_index] * 100

            # Get the class label
            predicted_class = class_labels[predicted_class_index]

            # Convert confidence score to integer
            confidence = int(confidence_score)

            if confidence < 50:
                error_message = "The uploaded image may not be clear. Please try again with a better, clear or related image."
                return render(request, 'index.html', {
                    'error_message': error_message
                })

            if predicted_class:
                image_url_1 = f"images/{predicted_class}/{predicted_class}_{np.random.randint(1, 6)}.jpg"
                image_url_2 = f"images/{predicted_class}/{predicted_class}_{np.random.randint(6, 11)}.jpg"
                result = FruitsData.objects.get(fruit=predicted_class)

    try:
        return render(request, 'index.html', {
            'result': result,
            'fruit_name': result.fruit_name.capitalize(),  # type: ignore
            'english_description': result.english_desc,  # type: ignore
            'in_urdu': result.in_urdu,  # type: ignore
            'urdu_description': result.urdu_desc,  # type: ignore
            'image_url_1': image_url_1,
            'image_url_2': image_url_2
        })
    # except:
    #     return render(request, 'index.html', {
    #         'error_message': "Something went wrong. Please try again."
    #     })
    except:
        return render(request, 'index.html')
