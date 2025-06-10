# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

# def training(request):
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import warnings
from django.shortcuts import render
from tensorflow.keras.utils import plot_model
from django.conf import settings

warnings.filterwarnings('ignore')

from django.conf import settings
from django.shortcuts import render
import os
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import EfficientNetV2B1

# def training(request):
#     """
#     Train a garbage classification model using EfficientNet and TensorFlow.
#     This function loads images from a dataset directory, preprocesses them, 
#     trains a deep learning model, and returns training metrics.
#     """
    
#     # Load Dataset
#     print("Loading dataset...")  # Add print statement
#     image_data = os.path.join(settings.MEDIA_ROOT, 'garbage_classification')
    
#     if not os.path.exists(image_data):
#         print(f"Dataset directory not found at: {image_data}")  # Add print statement
#         return render(request, 'users/training.html', {'error': 'Dataset directory not found!'})
    
#     print("Dataset found, loading image files...")  # Add print statement
#     files = [i for i in glob.glob(image_data + "//*//*")]
#     np.random.shuffle(files)
    
#     # Extract Labels
#     print("Extracting labels...")  # Add print statement
#     labels = [os.path.dirname(i).split("/")[-1] for i in files]
#     dataframe = pd.DataFrame(zip(files, labels), columns=["Image", "Label"])
    
#     # Count Classes
#     print("Counting classes...")  # Add print statement
#     sns.countplot(x=dataframe["Label"])
#     plt.xticks(rotation=50)
#     label_distribution_path = os.path.join(settings.MEDIA_ROOT, 'label_distribution.png')
#     plt.savefig(label_distribution_path)
#     plt.close()

#     # Define Training Parameters
#     print("Defining training parameters...")  # Add print statement
#     batch_size = 128
#     target_size = (224, 224)
#     validation_split = 0.2
    
#     # Load Image Dataset
#     print("Loading training and validation datasets...")  # Add print statement
#     train_data = tf.keras.preprocessing.image_dataset_from_directory(
#         image_data,
#         validation_split=validation_split,
#         subset="training",
#         seed=50,
#         image_size=target_size,
#         batch_size=batch_size,
#         label_mode="int"  # Ensure integer labels
#     )
#     val_data = tf.keras.preprocessing.image_dataset_from_directory(
#         image_data,
#         validation_split=validation_split,
#         subset="validation",
#         seed=100,
#         image_size=target_size,
#         batch_size=batch_size,
#         label_mode="int"
#     )

#     # Get Class Names & Number of Classes
#     print("Getting class names and number of classes...")  # Add print statement
#     class_names = train_data.class_names
#     num_classes = len(class_names)
    
#     print(f"Detected {num_classes} classes: {class_names}")  # Add print statement

#     # Show Sample Images
#     print("Showing sample images...")  # Add print statement
#     plt.figure(figsize=(10, 10))
#     for images, labels in train_data.take(1):
#         for i in range(8):
#             ax = plt.subplot(4, 4, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(class_names[labels[i]])
#             plt.axis("off")
#     sample_images_path = os.path.join(settings.MEDIA_ROOT, 'sample_images.png')
#     plt.savefig(sample_images_path)
#     plt.close()

#     # Load Pretrained Model (EfficientNetV2B1)
#     print("Loading pretrained model...")  # Add print statement
#     base_model = EfficientNetV2B1(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
#     base_model.trainable = False

#     # Build Model
#     print("Building model...")  # Add print statement
#     keras_model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax')  # Dynamically setting class count
#     ])

#     # Compile Model
#     print("Compiling model...")  # Add print statement
#     keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     print("Model compiled successfully.")  # Add print statement

#     # Train Model
#     print("Training model...")  # Add print statement
#     history = keras_model.fit(train_data, epochs=1, validation_data=val_data)
#     print("Training completed.")  # Add print statement

#     # Save Model
#     print("Saving model...")  # Add print statement
#     keras_model.save(os.path.join(settings.MEDIA_ROOT, "Garbage_model.h5"))
#     print("Model saved to disk.")  # Add print statement

#     # Evaluate Model
#     print("Evaluating model...")  # Add print statement
#     score, acc = keras_model.evaluate(val_data)
#     print(f'Test Loss: {score:.4f}, Test Accuracy: {acc:.4f}')  # Add print statement

#     # Save Training Metrics
#     print("Saving training metrics...")  # Add print statement
#     hist_df = pd.DataFrame(history.history)
#     train_loss = hist_df['loss'].tolist()
#     val_loss = hist_df['val_loss'].tolist()
#     train_accuracy = hist_df['accuracy'].tolist()
#     val_accuracy = hist_df['val_accuracy'].tolist()

#     print(f"Train Loss: {train_loss}")  # Add print statement
#     print(f"Validation Loss: {val_loss}")  # Add print statement
#     print(f"Train Accuracy: {train_accuracy}")  # Add print statement
#     print(f"Validation Accuracy: {val_accuracy}")  # Add print statement

#     # Plot Loss & Accuracy
#     print("Plotting loss and accuracy graphs...")  # Add print statement
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='Train Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.title('Train Loss & Validation Loss', fontsize=20)
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(train_accuracy, label='Train Accuracy')
#     plt.plot(val_accuracy, label='Validation Accuracy')
#     plt.title('Train Accuracy & Validation Accuracy', fontsize=20)
#     plt.legend()
    
#     training_performance_path = os.path.join(settings.MEDIA_ROOT, 'training_performance.png')
#     plt.savefig(training_performance_path)
#     plt.close()

#     print("Training performance plotted and saved.")  # Add print statement

#     # Render Results
#     print("Rendering results...")  # Add print statement
#     return render(request, 'users/training.html', {
#         'train_loss': train_loss,
#         'val_loss': val_loss,
#         'train_accuracy': train_accuracy,
#         'val_accuracy': val_accuracy,
#         'test_accuracy': acc,
#         'test_loss': score,
#         'num_classes': num_classes,
#         'class_names': class_names,
#         'label_distribution_path': label_distribution_path,
#         'sample_images_path': sample_images_path,
#         'training_performance_path': training_performance_path
#     })


import json
import os

def save_results_to_json(results, json_path):
    """
    Saves the training results to a JSON file.
    """
    with open(json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
        print(f"Results saved to {json_path}")

def load_results_from_json(json_path):
    """
    Loads the training results from a JSON file.
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            return json.load(json_file)
    return None


def training(request):
    """
    Train a garbage classification model using EfficientNet and TensorFlow.
    This function loads images from a dataset directory, preprocesses them, 
    trains a deep learning model, and returns training metrics.
    """

    json_results_path = os.path.join(settings.MEDIA_ROOT, 'training_results.json')
    results = load_results_from_json(json_results_path)

    if results:  # If results exist in the JSON file, return the cached results
        print("Loading results from JSON file...")
        return render(request, 'users/training.html', results)

    # Otherwise, proceed with training
    print("Loading dataset...")
    image_data = os.path.join(settings.MEDIA_ROOT, 'garbage_classification')
    
    if not os.path.exists(image_data):
        print(f"Dataset directory not found at: {image_data}")
        return render(request, 'users/training.html', {'error': 'Dataset directory not found!'})
    
    print("Dataset found, loading image files...")
    files = [i for i in glob.glob(image_data + "//*//*")]
    np.random.shuffle(files)
    
    # Extract Labels
    print("Extracting labels...")
    labels = [os.path.dirname(i).split("/")[-1] for i in files]
    dataframe = pd.DataFrame(zip(files, labels), columns=["Image", "Label"])
    
    # Count Classes
    print("Counting classes...")
    sns.countplot(x=dataframe["Label"])
    plt.xticks(rotation=50)
    label_distribution_path = os.path.join(settings.MEDIA_ROOT, 'label_distribution.png')
    plt.savefig(label_distribution_path)
    plt.close()

    # Define Training Parameters
    print("Defining training parameters...")
    batch_size = 128
    target_size = (224, 224)
    validation_split = 0.2
    
    # Load Image Dataset
    print("Loading training and validation datasets...")
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        image_data,
        validation_split=validation_split,
        subset="training",
        seed=50,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="int"
    )
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        image_data,
        validation_split=validation_split,
        subset="validation",
        seed=100,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # Get Class Names & Number of Classes
    print("Getting class names and number of classes...")
    class_names = train_data.class_names
    num_classes = len(class_names)
    
    print(f"Detected {num_classes} classes: {class_names}")

    # Show Sample Images
    print("Showing sample images...")
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(8):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    sample_images_path = os.path.join(settings.MEDIA_ROOT, 'sample_images.png')
    plt.savefig(sample_images_path)
    plt.close()

    # Load Pretrained Model (EfficientNetV2B1)
    print("Loading pretrained model...")
    base_model = EfficientNetV2B1(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Build Model
    print("Building model...")
    keras_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile Model
    print("Compiling model...")
    keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Model compiled successfully.")

    # Train Model
    print("Training model...")
    history = keras_model.fit(train_data, epochs=1, validation_data=val_data)
    print("Training completed.")

    # Save Model
    print("Saving model...")
    keras_model.save(os.path.join(settings.MEDIA_ROOT, "Garbage_model.h5"))
    print("Model saved to disk.")

    # Evaluate Model
    print("Evaluating model...")
    score, acc = keras_model.evaluate(val_data)
    print(f'Test Loss: {score:.4f}, Test Accuracy: {acc:.4f}')

    # Save Training Metrics
    print("Saving training metrics...")
    hist_df = pd.DataFrame(history.history)
    train_loss = hist_df['loss'].tolist()
    val_loss = hist_df['val_loss'].tolist()
    train_accuracy = hist_df['accuracy'].tolist()
    val_accuracy = hist_df['val_accuracy'].tolist()

    # Plot Loss & Accuracy
    print("Plotting loss and accuracy graphs...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Train Loss & Validation Loss', fontsize=20)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Train Accuracy & Validation Accuracy', fontsize=20)
    plt.legend()
    
    training_performance_path = os.path.join(settings.MEDIA_ROOT, 'training_performance.png')
    plt.savefig(training_performance_path)
    plt.close()

    print("Training performance plotted and saved.")

    # Save Results to JSON
    results = {
        'MEDIA_URL': settings.MEDIA_URL,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': acc,
        'test_loss': score,
        'num_classes': num_classes,
        'class_names': class_names,
        'label_distribution_path': settings.MEDIA_URL + 'label_distribution.png',
        'sample_images_path': settings.MEDIA_URL + 'sample_images.png',
        'training_performance_path': settings.MEDIA_URL + 'training_performance.png',
    }

    save_results_to_json(results, json_results_path)

    # Render Results
    print("Rendering results...")
    return render(request, 'users/training.html', results)



    # import numpy as np
    # import pandas as pd
    # import os
    # import cv2
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import tensorflow as tf
    # import keras
    # from tqdm import tqdm
    # from keras.callbacks import EarlyStopping, ModelCheckpoint
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # from sklearn.metrics import classification_report
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import LabelEncoder
    # import glob
    # import pandas as pan
    # import matplotlib.pyplot as plotter
    # import warnings
    # warnings.filterwarnings('ignore')
    # import os

    # # Create Files_Name
    # image_data = image_data = os.path.join(settings.MEDIA_ROOT, 'garbage_classification')
    # pd.DataFrame(os.listdir(image_data), columns=['Files_Name'])

    # files = [i for i in glob.glob(image_data + "//*//*")]
    # np.random.shuffle(files)
    # labels = [os.path.dirname(i).split("/")[-1] for i in files]
    # data = zip(files, labels)
    # dataframe = pan.DataFrame(data, columns=["Image", "Label"])
    # dataframe

    # sns.countplot(x=dataframe["Label"])
    # plotter.xticks(rotation=50)

    # train_data_dir = image_data
    # batch_size = 128
    # target_size = (224, 224)
    # validation_split = 0.2

    # train = tf.keras.preprocessing.image_dataset_from_directory(
    #     train_data_dir,
    #     validation_split=validation_split,
    #     subset="training",
    #     seed=50,
    #     image_size=target_size,
    #     batch_size=batch_size,
    # )
    # validation = tf.keras.preprocessing.image_dataset_from_directory(
    #     train_data_dir,
    #     validation_split=validation_split,
    #     subset="validation",
    #     seed=100,
    #     image_size=target_size,
    #     batch_size=batch_size,
    # )

    # class_names = train.class_names
    # class_names

    # plt.figure(figsize=(10, 10))
    # for images, labels in train.take(1):
    #     for i in range(8):
    #         ax = plt.subplot(8, 4, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")

    # base_model = tf.keras.applications.EfficientNetV2B1(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # base_model.trainable = False
    # keras_model = keras.models.Sequential()
    # keras_model.add(base_model)
    # keras_model.add(keras.layers.Flatten())
    # keras_model.add(keras.layers.Dropout(0.5))
    # keras_model.add(keras.layers.Dense(12, activation=tf.nn.softmax))  # 12 classes
    # keras_model.summary()
    



    # from tensorflow.keras.utils import plot_model 
    # plot_model(keras_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, dpi=100)  

    # # tf.keras.utils.plot_model(keras_model, to_file='model.png', show_shapes=True, show_layer_names=True, show_dtype=True, dpi=100)

    # checkpoint = ModelCheckpoint("my_keras_model.h5", save_best_only=True)

    # early_stopping = EarlyStopping(patience=10, restore_best_weights=True)  # patience from 5 to 10

    # keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # hist = keras_model.fit(train, epochs=10, validation_data=validation, callbacks=[checkpoint, early_stopping])

    # # Save the model
    # keras_model.save("Garbage_model.h5")

    # score, acc = keras_model.evaluate(validation)
    # print('Test Loss =', score)
    # print('Test Accuracy =', acc)

    # hist_df = pd.DataFrame(hist.history)
    # hist_df

    # train_loss = hist_df['loss'].tolist()
    # val_loss = hist_df['val_loss'].tolist()
    # train_accuracy = hist_df['accuracy'].tolist()
    # val_accuracy = hist_df['val_accuracy'].tolist()

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_loss, label='Train Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.title('Train Loss & Validation Loss', fontsize=20)
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracy, label='Train Accuracy')
    # plt.plot(val_accuracy, label='Validation Accuracy')
    # plt.title('Train Accuracy & Validation Accuracy', fontsize=20)
    # plt.legend()

    # return render(request, 'users/training.html', {'train_loss': train_loss,'val_loss': val_loss,'train_accuracy': train_accuracy,'val_accuracy': val_accuracy})

from django.shortcuts import render
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

# Load the model globally to avoid loading it every time the view is called
model_path = os.path.join(settings.MEDIA_ROOT, 'Garbage_model.h5')
model = load_model(model_path)

class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']  # Replace with your actual class names

def predict_Garbage(request):
    if request.method == 'POST' and request.FILES['file']:
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/garbage_classification/test_data")
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/garbage_classification/test_data/" + filename
        path = os.path.join(settings.MEDIA_ROOT, 'garbage_classification/test_data', filename)

        # Preprocess the image
        img_height, img_width = 224, 224
        img = image.load_img(path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        result = f"The uploaded image is predicted to be: {predicted_label}"

        # Display the image
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.title(f'Prediction: {predicted_label}')
        plt.axis('off')
        prediction_image_path = os.path.join(settings.MEDIA_ROOT, 'prediction_result.png')
        plt.savefig(prediction_image_path)
        prediction_image_url = "/media/prediction_result.png"

        return render(request, "users/UploadForm.html", {'path': uploaded_file_url, 'result': result, 'prediction_image_url': prediction_image_url})
    return render(request, "users/UploadForm.html")


