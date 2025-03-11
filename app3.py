import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import Swin_T_Weights, vit_b_16
import timm
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.linear_model import LogisticRegression

# Load the saved models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

model_vit = vit_b_16(pretrained=False).to(device)
model_vit.heads.head = torch.nn.Linear(model_vit.heads.head.in_features, num_classes)
model_vit.load_state_dict(torch.load('trained_vit_model.pth', map_location=device))
model_swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
model_swin.head = torch.nn.Linear(model_swin.head.in_features, 7)
model_deit = timm.create_model('deit_small_patch16_224', pretrained=False).to(device)
model_deit.head = torch.nn.Linear(model_deit.head.in_features, num_classes)

model_vit.load_state_dict(torch.load('trained_vit_model.pth', map_location=device))
model_swin.load_state_dict(torch.load('Trained_SWIN_T_model_new.pth', map_location=device))
model_deit.load_state_dict(torch.load('trained_deit_s_model.pth', map_location=device))

model_vit.eval()
model_swin.eval()
model_deit.eval()

# Load the stacking meta-learner
with open('stacking_meta_learner_xgboost.pkl', 'rb') as f:
    loaded_meta_learner = pickle.load(f)

# Define transformations
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

# Class labels
class_labels = ['10 ETB', '100 ETB', '200 ETB', '5 ETB', '50 ETB','Fake 100 ETB','Fake 200 ETB']

# Streamlit app
st.title("Ethiopian Banknote denomination and counterfeit detection using Vision transformer")
# Initialize session state for image and prediction results
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
# Option to capture image from camera
if st.button("Capture Image"):
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise IOError("Cannot open webcam")
        ret, frame = camera.read()
        cv2.imwrite("captured_image.jpg", frame)
        camera.release()
        cv2.destroyAllWindows()

        image = Image.open("captured_image.jpg").convert('RGB')
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        image_tensor = transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            vit_output = model_vit(image_tensor)
            swin_output = model_swin(image_tensor)
            deit_output = model_deit(image_tensor)
        stacked_predictions = np.concatenate((vit_output.cpu().numpy(), swin_output.cpu().numpy(), deit_output.cpu().numpy()), axis=1)
        final_predictions = loaded_meta_learner.predict_proba(stacked_predictions)

        # Banknote detection check
        if np.max(final_predictions) < 0.6:
            st.warning("Banknote not detected. Please try again with a clearer image of a banknote.")
        else:
            st.subheader("Prediction Results:")
            for label, probability in zip(class_labels, final_predictions[0]):
                st.write(f'{label}: {probability:.2f}')
            predicted_class_index = np.argmax(final_predictions)
            predicted_class = class_labels[predicted_class_index]
            st.write(f"Predicted Class: {predicted_class}")
            st.session_state.prediction_result = final_predictions
            st.session_state.image_result = image

    except Exception as e:
        st.write(f"Error capturing image: {e}")

# Option to upload image from file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vit_output = model_vit(image_tensor)
        swin_output = model_swin(image_tensor)
        deit_output = model_deit(image_tensor)

    stacked_predictions = np.concatenate((vit_output.cpu().numpy(), swin_output.cpu().numpy(), deit_output.cpu().numpy()), axis=1)
    final_predictions = loaded_meta_learner.predict_proba(stacked_predictions)
    st.session_state.prediction_result = final_predictions
    st.session_state.image_result = image

    # Banknote detection check
    if np.max(final_predictions) < 0.6:
        st.warning("Banknote not detected. Please try again with a clearer image of a banknote.")
    else:
        st.subheader("Prediction Results:")
        for label, probability in zip(class_labels, final_predictions[0]):
            st.write(f'{label}: {probability:.2f}')
        predicted_class_index = np.argmax(final_predictions)
        predicted_class = class_labels[predicted_class_index]
        st.write(f"Predicted Class: {predicted_class}")
if st.button("Clear Prediction"):
    st.session_state.prediction_result = None  # Clear prediction results
    st.session_state.image_result = None       # Clear displayed image
