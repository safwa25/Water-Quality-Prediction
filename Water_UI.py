import os
import streamlit as st
import requests
from PIL.Image import Image
from streamlit_lottie import st_lottie
import joblib
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data_cleaned=pd.read_csv(r'C:\Users\Safwa\Downloads\Ai_project\water_potability_cleaned2.csv')
X = data_cleaned.drop(columns=['Potability'])
y=data_cleaned['Potability']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#--------------------------------------------------------------------------------------------------------------------
icon=Image.open(r'C:\Users\Safwa\Downloads\Ai_project\home.png')
st.set_page_config(page_title='Water Quality',  page_icon='💧')
#Side Bar
#--------------------------------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(icon)


st.sidebar.write("""
    <h1 style="text-align:center; color: rgb(242, 146, 0);"> Choose one of the models</h1>
""", unsafe_allow_html=True)

st.sidebar.write('---')
with st.sidebar.container():
    Model=st.selectbox('Select Ai model you want to apply :',('Logestic Regression','Support Vector Machine','Decision Tree','Random Forest'))
#------------------------------------------------------------------------------------------------------------------------------------------------
#Load Models
#------------------------------------------------------------------------------------------------------------------------------------------------

Logistic = joblib.load(open('Logistic__model', 'rb'))
SVM=joblib.load(open('SVM__model', 'rb'))
DecisionT=joblib.load(open('descionTree__model', 'rb'))
RandomForest=joblib.load(open('randomForest__model', 'rb'))

#-----------------------------------------------------------------------------------------------------------------------------------------
#Logestic Regression
logist = LogisticRegression(random_state = 0)
logist.fit(X_train, y_train)
yl_pred = logist.predict(X_test)
cr=classification_report(y_test, yl_pred)
cml = confusion_matrix(y_test, yl_pred)
#------------------------------------------------------------------------------------------------------------------------------------------
#Random Forest
rf = RandomForestClassifier(n_estimators = 12, criterion = 'gini', random_state = 42)
rf.fit(X_train, y_train)
yrf_pred = rf.predict(X_test)
crf=classification_report(y_test, yrf_pred)
cmrf = confusion_matrix(y_test, yrf_pred)
#------------------------------------------------------------------------------------------------------------------------------------------
#Decision tree
decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=14, random_state = 42)
decisionTree.fit(X_train, y_train)
ytree_pred = decisionTree.predict(X_test)
crtree=classification_report(y_test,ytree_pred)
cmtree=confusion_matrix(y_test, ytree_pred)
#------------------------------------------------------------------------------------------------------------------------------------------
#SVM
support=svm.SVC()
support.fit(X_train,y_train)
ysvm_pred = support.predict(X_test)
crsvm=classification_report(y_test,ysvm_pred)
cmsvm=confusion_matrix(y_test, ysvm_pred)
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------
#Home page
#-------------------------------------------------------------------------------------------------------------------------------------------------

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def prepare_input_data_for_model(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    # Input data
    A = [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]

    # Mean and standard deviation values obtained from your notebook
    mean_values = [7.080855056957347, 196.39242287662802, 21957.112200016294, 7.121793997984291, 333.7851233506843, 426.1299736366093, 14.28346155555034, 66.39567132362379, 3.966612088814371]
    std_dev_values = [1.4699575575433586,32.01718932405032,8592.820396897496,1.5441262126241226,36.14570094419796,80.56414377902169,3.2883667161960286,15.76990111741393,0.7764085970881656]

    # Convert the lists to numpy arrays
    mean = np.array(mean_values)
    std_dev = np.array(std_dev_values)

    # Store the input values in a numpy array
    sample = np.array(A)

    # Apply standard scaling using pre-calculated mean and standard deviation values
    scaled_sample = (sample - mean) / std_dev

    # Reshape the scaled_sample to match the shape of the input sample
    scaled_sample = scaled_sample.reshape(1, -1)

    return scaled_sample

st.write("""
    <h1 style="text-align:center; color: rgb(6, 75, 231);">AI-Driven Water Quality Prediction</h1>
    <h1 style="text-align:center; color: rgb(6, 75, 231);">Ensuring Safe Drinking Water 🚰</h1>
""", unsafe_allow_html=True)

st.write('---')
Image = Image.open(r'C:\Users\Safwa\Downloads\Ai_project\Blue Illustrative World Water Day Instagram Post.png')
st.image(Image)
lottie_link = "https://lottie.host/d78f2c63-c4c6-48e1-a830-38c1becbd49e/p34VrQBnN6.json"
animation = load_lottie(lottie_link)

st.write('---')
st.subheader("Implementing Advanced Predictive Models for Comprehensive Assessment of Drinking Water Quality")
with st.container():
    right_column, left_column = st.columns(2)

    with right_column:
        ph= st.number_input('Potential of hydrogen(pH): ', min_value=0.0, max_value=14.0, value=0.0, step=0.1)

        Hardness = st.number_input('Hardness: ', min_value=117.125160, max_value=277.0,value=120.0, step=0.1)

        Solids= st.number_input('Solids:', min_value=320.942611, max_value=44831.869873, value=400.0,step=0.1)

        Chloramines= st.number_input('Chloramines :', min_value=3.146221, max_value=11.096086, value=4.0, step=0.1)

        Sulfate= st.number_input('Sulfate: ', min_value=129.000000, max_value=481.030642, value=140.0, step=0.1)

        Conductivity = st.number_input('Conductivity: ', min_value=191.647579, max_value= 655.879140, value=200.0, step=0.1)

        Organic_carbon = st.number_input('Organic Carbon: ', min_value=5.328026, max_value=23.295427,value=6.0, step=0.01)

        Trihalomethanes= st.number_input('Trihalomethanes: ', min_value=0.738000, max_value=124.000000,value=8.0, step=0.01)

        Turbidity= st.number_input('Turbidity: ', min_value=1.848797,max_value=6.091233, value=2.0, step=0.01)

        sample = prepare_input_data_for_model(ph,Hardness, Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity)

    with left_column:
        st_lottie(animation, speed=1, height=400, key="initial")
#____________________________________________________________________________________________________________________________

    if st.button('Predict'):
        if Model=="Logestic Regression":
            pred_Y=Logistic.predict(sample)
            if pred_Y == 0:
                st.write("""
                <h3 style="color: rgb(242, 146, 0);">This is a non potable water</h3>
                """, unsafe_allow_html=True)
            else:
                st.write("""
                <h3 style=" color: rgb(242, 146, 0);">This is a potable water</h3>
                """, unsafe_allow_html=True)
            st.write('---')
            st.write("""
            <h3 style=" color: rgb(6, 75, 231);">Classification Report Of Logistic Regression </h3>
            """, unsafe_allow_html=True)
            st.text(cr)
            st.write('---')
            fig, ax = plt.subplots()
            sns.heatmap(cml, annot=True, cmap="Blues", fmt="d")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            # Display confusion matrix
            st.write("""
                        <h3 style=" color: rgb(6, 75, 231);">Confusion Matrix Of Logistic Regression</h3>
                        """, unsafe_allow_html=True)
            st.pyplot(fig)

#------------------------------------------------------------------------------------------------------------------------------
        elif Model=="Random Forest":
            pred_Y = RandomForest.predict(sample)
            if pred_Y == 0:
                st.write("""
                            <h3 style="color: rgb(242, 146, 0);">This is a non potable water</h3>
                            """, unsafe_allow_html=True)
            else:
                st.write("""
                            <h3 style=" color: rgb(242, 146, 0);">This is a potable water</h3>
                            """, unsafe_allow_html=True)
            st.write('---')
            st.write("""
                        <h3 style=" color: rgb(6, 75, 231);">Classification Report Of Random Forest </h3>
                        """, unsafe_allow_html=True)
            st.text(crf)
            st.write('---')
            fig, ax = plt.subplots()
            sns.heatmap(cmrf, annot=True, cmap="Blues", fmt="d")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            # Display confusion matrix
            st.write("""
                                    <h3 style=" color: rgb(6, 75, 231);">Confusion Matrix Of Random Forest</h3>
                                    """, unsafe_allow_html=True)
            st.pyplot(fig)
#--------------------------------------------------------------------------------------------------------------------------------
        elif Model=="Decision Tree":
            pred_Y = DecisionT.predict(sample)
            if pred_Y == 0:
                st.write("""
                                        <h3 style="color: rgb(242, 146, 0);">This is a non potable water</h3>
                                        """, unsafe_allow_html=True)
            else:
                st.write("""
                                        <h3 style=" color: rgb(242, 146, 0);">This is a potable water</h3>
                                        """, unsafe_allow_html=True)
            st.write('---')
            st.write("""
                                    <h3 style=" color: rgb(6, 75, 231);">Classification Report Of Decision Tree </h3>
                                    """, unsafe_allow_html=True)
            st.text(crtree)
            st.write('---')
            fig, ax = plt.subplots()
            sns.heatmap(cmtree, annot=True, cmap="Blues", fmt="d")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            # Display confusion matrix
            st.write("""
                                                <h3 style=" color: rgb(6, 75, 231);">Confusion Matrix Of Decision Tree</h3>
                                                """, unsafe_allow_html=True)
            st.pyplot(fig)

#-----------------------------------------------------------------------------------------------------------------------------------
        else:
            pred_Y = SVM.predict(sample)
            if pred_Y == 0:
                st.write("""
                                        <h3 style="color: rgb(242, 146, 0);">This is a non potable water</h3>
                                        """, unsafe_allow_html=True)
            else:
                st.write("""
                                        <h3 style=" color: rgb(242, 146, 0);">This is a potable water</h3>
                                        """, unsafe_allow_html=True)
            st.write('---')
            st.write("""
                                    <h3 style=" color: rgb(6, 75, 231);">Classification Report Of Support Vector Machine</h3>
                                    """, unsafe_allow_html=True)
            st.text(crsvm)
            st.write('---')
            fig, ax = plt.subplots()
            sns.heatmap(cmsvm, annot=True, cmap="Blues", fmt="d")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            # Display confusion matrix
            st.write("""
                                                <h3 style=" color: rgb(6, 75, 231);">Confusion Matrix Of Support Vector Machine</h3>
                                                """, unsafe_allow_html=True)
            st.pyplot(fig)


