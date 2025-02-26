# Diagnostic-tool

This repository contains machine learning models for early detection of common medical conditions. The models included are:
Diabetes Detector: Uses a trained machine learning model to predict diabetes based on patient data.
Heart Disease Detector: Predicts the likelihood of heart disease based on provided risk factors.
Pneumonia Detector: A deep learning model to detect pneumonia from medical images.
    ```
    
    │── app.py                        # Python application script
    │── diabetes.csv                   # Dataset for diabetes detection
    │── Diabetes_detector.ipynb         # Jupyter notebook for diabetes model training
    │── diabetes_model.pkl              # Trained diabetes model (Pickle format)
    │── diabetes_scaler.pkl             # Scaler for preprocessing diabetes data
    │── heart_disease_data.csv          # Dataset for heart disease detection
    │── heart_disease_detector.ipynb    # Jupyter notebook for heart disease model training
    │── Heart_disease_model.pkl         # Trained heart disease model (Pickle format)
    │── pneumonia_detection.ipynb       # Jupyter notebook for pneumonia detection model
    │── pneumonia_detector.h5           # Trained pneumonia model (H5 format)

1. Clone the repository:

       git clone https://github.com/TanishaChauhan19/Diagnostic-tool.git
       cd Diagnostic-tool

2.Intall the dependenices

3. Run the app
    ```
   streamlit run app.py

Here’s a screenshot of the app:

  ![Image Description](https://github.com/TanishaChauhan19/Diagnostic-tool/blob/main/app%20img.jpg?raw=true)
  ![Image Description](https://github.com/TanishaChauhan19/Diagnostic-tool/blob/main/app2%20img.jpg?raw=true)
Usage Instructions

1.Diabetes & Heart Disease Prediction
Upload or input patient data (CSV format).
The model will predict the likelihood of diabetes or heart disease.

2.Pneumonia Detection
Upload a chest X-ray image.
The deep learning model (pneumonia_detector.h5) will classify it as normal or pneumonia-positive.


