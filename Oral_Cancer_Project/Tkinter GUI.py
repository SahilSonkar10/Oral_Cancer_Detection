import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
model = tf.keras.models.load_model(r"D:\keras model\cancer prediction model\save_model\new_model.keras")

# Default image size
DEFAULT_IMAGE_SIZE = (300, 300)

# Function to preprocess the image
def preprocess_image(image_path):
    img_width, img_height = 64, 64
    image = Image.open(image_path)
    image = image.resize((img_width, img_height), resample=Image.BILINEAR)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_image():
    file_path = selected_image.get()
    if file_path:
        try:
            preprocessed_image = preprocess_image(file_path)
            prediction = model.predict(preprocessed_image)
            probability = prediction.flatten()[0]
            prediction_label = "Benign" if probability < 0.5 else "Malignant"
            result_label.config(text=f"Detection done successfully.\nPrediction: {prediction_label}",
                                fg="green" if prediction_label == "Benign" else "red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("Warning", "Please select an image.")

# Function to reset the displayed image and prediction result
def reset_image():
    selected_image.set('')  # Clear the selected image
    image_label.config(image='')
    result_label.config(text='')

# Function to display the selected image immediately after selection
def display_selected_image():
    file_path = selected_image.get()
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail(DEFAULT_IMAGE_SIZE)  # Ensure image fits within default size
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
            image_label.photo = photo  # Maintain reference to the PhotoImage object
            root.update_idletasks()  # Update the window to fit the new image size
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("Warning", "Please select an image.")

# Function to handle zoom in with mouse scroll
def zoom_in_mouse(event):
    global DEFAULT_IMAGE_SIZE
    DEFAULT_IMAGE_SIZE = (int(DEFAULT_IMAGE_SIZE[0] * 1.1), int(DEFAULT_IMAGE_SIZE[1] * 1.1))
    display_selected_image()

# Function to handle zoom out with mouse scroll
def zoom_out_mouse(event):
    global DEFAULT_IMAGE_SIZE
    DEFAULT_IMAGE_SIZE = (int(DEFAULT_IMAGE_SIZE[0] * 0.9), int(DEFAULT_IMAGE_SIZE[1] * 0.9))
    display_selected_image()

# Function to handle zoom in with keyboard shortcut
def zoom_in_keyboard(event):
    global DEFAULT_IMAGE_SIZE
    DEFAULT_IMAGE_SIZE = (int(DEFAULT_IMAGE_SIZE[0] * 1.1), int(DEFAULT_IMAGE_SIZE[1] * 1.1))
    display_selected_image()

# Function to handle zoom out with keyboard shortcut
def zoom_out_keyboard(event):
    global DEFAULT_IMAGE_SIZE
    DEFAULT_IMAGE_SIZE = (int(DEFAULT_IMAGE_SIZE[0] * 0.9), int(DEFAULT_IMAGE_SIZE[1] * 0.9))
    display_selected_image()

# Create the main window
root = tk.Tk()
root.title("Oral Cancer Detection")

# Load and display logo image
logo_image = Image.open(r"D:\keras model\cancer prediction model\save_model\sgsitslogo.jpg")  # Adjust the path accordingly
logo_image = logo_image.resize((300, 200), Image.ANTIALIAS)  
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo_photo)
logo_label.image = logo_photo
logo_label.pack()

# Create and place widgets
title_label1 = tk.Label(root, text="Oral Cancer Prediction", font=('Helvetica', 20), fg="black")
title_label1.config(font=("Arial", 20))  # Change font style
title_label1.pack(pady=(20, 10))

title_label2 = tk.Label(root, text="Using Deep Learning (Convolutional Neural Network)", font=('Helvetica', 16),
                        fg="black")
title_label2.config(font=("Arial", 12))  # Change font style
title_label2.pack(pady=(0, 20), padx=120)

choose_button = tk.Button(root, text="Choose Image", font=('Helvetica', 14),
                          command=lambda: [selected_image.set(filedialog.askopenfilename()), display_selected_image()])
choose_button.pack(pady=10)

predict_button = tk.Button(root, text="Predict", font=('Helvetica', 14), command=predict_image)
predict_button.pack(pady=10)

reset_button = tk.Button(root, text="Reset", font=('Helvetica', 14), command=reset_image)
reset_button.pack(pady=10)

selected_image = tk.StringVar()  # Variable to store selected image path

# Label to display the selected image
image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=('Helvetica', 16))
result_label.pack(pady=20)

# Bind mouse scroll events for zooming
root.bind('<MouseWheel>', zoom_in_mouse)
root.bind('<Shift-MouseWheel>', zoom_out_mouse)  # Add Shift key modifier for zoom out

# Bind keyboard shortcuts for zooming
root.bind('<Control-plus>', zoom_in_keyboard)
root.bind('<Control-minus>', zoom_out_keyboard)

root.mainloop()
