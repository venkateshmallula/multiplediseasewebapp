# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 00:43:00 2022

@author: mallu
"""


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav','rb'))

breast_cancer_model = pickle.load(open('breast_cancer_model.sav','rb'))

kidney_disease_model = pickle.load(open('kidney_disease_model.sav','rb'))


# sidebar for navigation# sidebar for navigation
selected = option_menu(menu_title='Multiple Disease Prediction Web App',
                       options=['Home-Page','Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction','Breast Cancer Prediction','Kidney Disease Prediction'],
                       icons=['house','activity','heart','person','gender-female','droplet'],
                       default_index=0,
                       orientation='horizontal')

if (selected == 'Home-Page'):
 html_temp = """
 <div style ="background-color:yellow;padding:13px">
 <h1 style ="color:black;text-align:center;">Multiple Disease Prediction using Machine Learning and Streamlit </h1>
 </div>
 """
      
 # this line allows us to display the front end aspects we have 
 # defined in the above code
 st.markdown(html_temp, unsafe_allow_html = True)
 st.write(" ")
 st.write("   The Multiple Disease Prediction application is a machine learning-based project that aims to predict various diseases, including diabetes, heart disease, kidney disease, Parkinson's disease, and breast cancer. The application utilizes advanced machine learning algorithms such as Support Vector Machine (SVM), Logistic Regression, and TensorFlow with Keras.")
 st.write(" ")
 st.write("This application offers a user-friendly interface with five disease prediction options: heart disease, kidney disease, diabetes, Parkinson's disease, and breast cancer. Upon selecting a specific disease, the user is prompted to input the required parameters for the prediction model. Once the parameters are entered, the application utilizes the trained models to predict the disease outcome and presents the result to the user.")
 st.write(" ")
 st.write("The accuracy of the prediction models varies for each disease, with SVM achieving 78% accuracy for diabetes and 87% accuracy for Parkinson's disease, logistic regression achieving 85% accuracy for heart disease, and TensorFlow with Keras achieving 97% accuracy for kidney disease and 95% accuracy for breast cancer.")
  
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        
        st.write("")

    with col2:
        
       img = Image.open("d1.jpg")
       st.image(img,width=150)
    with col3:
        
        st.write("")
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            text-color: black;
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    add_bg_from_local('diabetes.jpg')
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
      if (
        Pregnancies.strip() == "" or Glucose.strip() == "" or BloodPressure.strip() == ""
        or SkinThickness.strip() == "" or Insulin.strip() == "" or BMI.strip() == ""
        or DiabetesPedigreeFunction.strip() == "" or Age.strip() == ""
      ):
        st.error("Please enter all the required parameters.")
      else:
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        
        st.write("")

    with col2:
        
        img = Image.open("h6.jpg")
        st.image(img,width=150)
    with col3:
        
        st.write("")
    #for adding Image
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            font-color: black;
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    add_bg_from_local('h1.jpg')
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
      if (
        age is None or sex is None or cp is None or trestbps is None or chol is None or fbs is None
        or restecg is None or thalach is None or exang is None or oldpeak is None or slope is None or ca is None or thal is None
      ):
        st.error("Please enter all the required parameters.")
      else:
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        st.success(heart_diagnosis)
      

  # Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
  
    col1, col2, col3 = st.columns(3)

    with col1:
        
        st.write("")

    with col2:
        
       img = Image.open("p1.jpg")
       st.image(img,width=150)
    with col3:
        
        st.write("")
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            text-color: black;
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    add_bg_from_local('p4.jpg')
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinsons Test Result"):
        if (
            fo.strip() == "" or fhi.strip() == "" or flo.strip() == ""
            or Jitter_percent.strip() == "" or Jitter_Abs.strip() == ""
            or RAP.strip() == "" or PPQ.strip() == "" or DDP.strip() == ""
            or Shimmer.strip() == "" or Shimmer_dB.strip() == "" or APQ3.strip() == ""
            or APQ5.strip() == "" or APQ.strip() == "" or DDA.strip() == "" or NHR.strip() == ""
            or HNR.strip() == "" or RPDE.strip() == "" or DFA.strip() == "" or spread1.strip() == ""
            or spread2.strip() == "" or D2.strip() == "" or PPE.strip() == ""
        ):
            st.error("Please enter all the required parameters.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                                Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                                                RPDE, DFA, spread1, spread2, D2, PPE]])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"

            st.success(parkinsons_diagnosis)
      
     
  # Breast cancer Prediction Page
if (selected == "Breast Cancer Prediction"):
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            text-color: black;
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    add_bg_from_local('b1.jpg')
    ########################################
    import numpy as np
    import pandas as pd
    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    tf.random.set_seed(3)
    from tensorflow import keras
    import streamlit as st

    # loading the data from sklearn
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

    # loading the data to a data frame
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

    # adding the 'target' column to the data frame
    data_frame['label'] = breast_cancer_dataset.target

    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']

    #splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    #Standardize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # importing tensorflow and Keras
    import tensorflow as tf
    tf.random.set_seed(3)
    from tensorflow import keras
    # setting up the layers of Neural Network

    model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
    ])
    # compiling the Neural Network

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # training the neural Network

    history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
    loss, accuracy = model.evaluate(X_test_std, Y_test)

    Y_pred = model.predict(X_test_std)

    # Set the page title
    st.title("Breast Cancer Prediction using Machine Learning")
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Breast Cancer Prediction ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    features_names = X.columns.tolist()
    if st.button('Show Feaatures'):
      st.write(features_names)

    input_value = st.text_input("Enter the features separated by Comma")

    input_list = input_value.split(',')
    # Create a button for prediction
    if st.button("Predict"):
      try:
             input_data = np.array([input_list], dtype=np.float32)
             st.write(input_data)

             # change the input_data to a numpy array
             input_data_as_numpy_array = np.asarray(input_data)

             # reshape the numpy array as we are predicting for one data point
             input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

             # standardizing the input data
             input_data_std = scaler.transform(input_data_reshaped)

             prediction = model.predict(input_data_std)

             prediction_label = [np.argmax(prediction)]

             if(prediction_label[0] == 0):
                   st.success('The tumor is Malignant')

             else:
                   st.success('The tumor is Benign')
      except ValueError:
                 st.error("Invalid input. Please enter numeric values for all features.")


    
  # Kidney disease Prediction Page
if (selected == "Kidney Disease Prediction"):
        import base64
        def add_bg_from_local(image_file):
            with open(image_file, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                st.markdown(
                f"""
                <style>
                .stApp {{
                background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                background-size: cover;
                text-color: black;
                }}
                </style>
                """,
                unsafe_allow_html=True
                )
        add_bg_from_local('kidney-1030x659.jpg')
    
    # page title
        import numpy as np
        import pandas as pd
        import os, sys
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # loading the dataset
        data = pd.read_csv('kidney_disease.csv')
        
        #imputing null values
        from sklearn.impute import SimpleImputer
        imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        
        df_imputed = pd.DataFrame(imp_mode.fit_transform(data))
        df_imputed.columns=data.columns
        
        
        
        df_imputed['classification']=df_imputed['classification'].apply(lambda x: 'ckd' if x=='ckd\t' else x)
        
        df_imputed['cad']=df_imputed['cad'].apply(lambda x: 'no' if x=='\tno' else x)
        
        df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'no' if x=='\tno' else x)
        df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'yes' if x=='\tyes' else x)
        df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'yes' if x==' yes' else x)
        
        df_imputed['rc']=df_imputed['rc'].apply(lambda x: '5.2' if x=='\t?' else x)
        
        df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\6200?' else x)
        df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\8400' else x)
        df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\t?' else x)
        
        df_imputed['pcv']=df_imputed['pcv'].apply(lambda x: '41' if x=='\t43' else x)
        df_imputed['pcv']=df_imputed['pcv'].apply(lambda x: '41' if x=='\t?' else x)
        
        temp=df_imputed ["classification"].value_counts()
        
        for i in data.select_dtypes (exclude=["object"]).columns:
          df_imputed[i]=df_imputed[i].apply(lambda x: float(x))
        
        # Label encoding to convert categorical values to numerical
        from sklearn import preprocessing
        df_enco=df_imputed.apply(preprocessing.LabelEncoder().fit_transform)
        
        # Lets make some final changes to the data
        # Seperate independent and dependent variables and drop the ID column
        x=df_enco.drop(["id","classification"],axis=1)
        y=df_enco["classification"]
        
        # Lets detect the Label balance
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter
        
        # Lets balance the Labels
        ros= RandomOverSampler()
        x_ros, y_ros = ros.fit_resample(x, y)
        
        #Initialize a MinMaxScaler and scale the features to between 1 and 1 to normalize them.
        #The MinMaxScaler transforms features by scaling them to a given range.
        #The fit_transform() method fits to the data and then transforms it. We don't need to scale the labels.
        #Scale the features to between -1 and 1
        # Scaling is important in the algorithms such as support vector machines (SVM) and k-nearest neighbors (KNN) where distance
        # between the data points is important.
        scaler = MinMaxScaler ((-1,1))
        x=scaler.fit_transform(x_ros)
        y=y_ros
        
        # Applying PCA
        # The code below has .95 for the number of components parameter.
        # It means that scikit-Learn choose the minimum number of principal components such that 95% of the variance is retained.
        from sklearn.decomposition import PCA
        pca = PCA(.95)
        X_PCA=pca.fit_transform(x)
        
        #Split the dataset
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(X_PCA, y, test_size=0.2, random_state=7)
        
        # importing tensorflow and Keras
        import tensorflow as tf 
        tf.random.set_seed(3)
        from tensorflow import keras
        
        # setting up the layers of Neural Network
        
        model = keras.Sequential([
                                  keras.layers.Flatten(input_shape=(18,)),
                                  keras.layers.Dense(9, activation='relu'),
                                  keras.layers.Dense(2, activation='sigmoid')
        ])
        
        # compiling the Neural Network
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # training the neural Network
        
        history = model.fit(x_train, y_train, validation_split=0.1, epochs=20)
        
        #Accuracy of the model on test data
        loss, accuracy = model.evaluate(x_test, y_test)
        
        Y_pred = model.predict(x_test)
        
        import streamlit as st
        
        # Set the page title
        st.title("Kidney Disease Prediction using Machine Learning")
        html_temp = """
        <div style ="background-color:yellow;padding:13px">
        <h1 style ="color:black;text-align:center;">Streamlit Kidney Disease Prediction ML App </h1>
        </div>
        """
        # this line allows us to display the front end aspects we have 
        # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
        features_names = ["age","blood pressure","specific gravity","albumin","sugar","red blood cells","pus cell","pus cell clumps","bacteria","blood glucose random","blood urea","serum creatinine","sodium","potassium","hemoglobin","packed cell volume","white blood cell count","red blood cell count","hypertension","diabetes mellitus","coronary artery disease","appetite","pedal edema","anemia"]
        if st.button('Show Feaatures'):
          st.write(features_names)
          
        input_value = st.text_input("Enter the features separated by Comma")
        
        input_list = input_value.split(',')
        if st.button("Predict"):
              try:
                  input_data = np.array([input_list], dtype=np.float32)
                
                  # change the input_data to a numpy array
                  input_data_as_numpy_array = np.asarray(input_data)
        
                  # reshape the numpy array as we are predicting for one data point
                  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
                  # standardizing the input data
                  input_data_std = scaler.transform(input_data_reshaped)
        
                  #feature extraction and dimensional reduction
                  input_data_std_PCA = pca.transform(input_data_std)
        
                  prediction = model.predict(input_data_std_PCA)
                 
        
                  prediction_label = [np.argmax(prediction)]
                 
        
                  if(prediction_label[0] == 0):
                      st.success('kidneys are not Infected')
        
                  else:
                      st.success('kidneys are Infected')
              except ValueError:
                         st.error("Invalid input. Please enter numeric values for all features.")

