import streamlit as st 
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
          height: 100%;

        /* Center and scale the image nicely */
           background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        background-color: grey;
        background-blend-mode: multiply;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)

add_bg_from_local('car.png')

st.title("CAR RECOMMENDATION SYSTEM")


select_Dataset=st.sidebar.selectbox("Select Dataset",options=["Select Dataset","Car"])
select_Activity=st.sidebar.selectbox("Select Activity",options=["Select Activity","Model","Recommendation"])

car_data=pd.read_csv('data.csv')
car_data = car_data.dropna(axis=0)
carMakeLst=[]#variable will contain list of all car brands that are there in the dataset
for Brand in (car_data.Make):
    if Brand not in carMakeLst:
        carMakeLst.append(Brand)

car_data['MakeInd']=[carMakeLst.index(Make) for Make in car_data.Make]

carFuelLst=[]#variable will contain list of all car brands that are there in the dataset
for fuel in (car_data['Engine Fuel Type']):
    if fuel not in carFuelLst:
        carFuelLst.append(fuel)

car_data['FuelInd']=[carFuelLst.index(fuel) for fuel in car_data['Engine Fuel Type']]

carCategory=[]#variable will contain list of all car categories that are there in the dataset
for Category in (car_data['Market Category']):
    if Category not in carCategory:
        carCategory.append(Category)
#print(carCategory)
car_data['MktCategoryInd']=[carCategory.index(Cat) for Cat in car_data['Market Category']]

carTTLst=[]#variable will contain list of all car transmission type that are there in the dataset
for Type in (car_data.TransmissionType):
    if Type not in carTTLst:
        carTTLst.append(Type)

car_data['TypeInd']=[carTTLst.index(Type) for Type in car_data.TransmissionType]

carDWLst=[]#variable will contain list of all car drive type that are there in the dataset
for Wheel in (car_data.DrivenWheels):
    if Wheel not in carDWLst:
        carDWLst.append(Wheel)

car_data['DWInd']=[carDWLst.index(Wheel) for Wheel in car_data.DrivenWheels]

carVSLst=[]#variable will contain list of all car size that are there in the dataset
for Size in (car_data.VehicleSize):
    if Size not in carVSLst:
        carVSLst.append(Size)

car_data['VSInd']=[carVSLst.index(Size) for Size in car_data.VehicleSize]

carVTLst=[]#variable will contain list of all car style that are there in the dataset
for Style in (car_data.VehicleStyle):
    if Style not in carVTLst:
        carVTLst.append(Style)

car_data['VTInd']=[carVTLst.index(Style) for Style in car_data.VehicleStyle]
new_feature_col=['MakeInd','TypeInd','DWInd','VSInd','VTInd','Engine HP','MktCategoryInd','FuelInd','Number of Doors','highway MPG','city mpg','Engine Cylinders']
X=car_data[new_feature_col]#will use only these features for training and ignore rest
y=car_data.Model

train_X, test_X, train_y, test_y = train_test_split(X,y, random_state = 1)

if(select_Activity=="Select Activity"):
    st.header("INTRODUCTION")
    st.write("The car recommendation system allows users to select their preferred car features by clicking on options prepared by the system. The system then compares the performance of three machine learning models, including random forest, support vector machine (SVM), and decision tree, to determine the best model for making recommendations. The model that performs the best is then used to provide car recommendations to the user based on their selected preferences.")
    st.subheader("THE DATASET")
    st.write("The dataset is acquired from the website Kaggle.com. Over 11,231 samples are available. The total number of columns or variables are 16. Below shown the dataset after preforming data cleaning.")
    if(select_Dataset=="Car"):
        car_data
    st.write("The chosen dataset is collected fromTraining and testing dataset will be automatically split from the original dataset at 80% for training and 20% for testing")
    
    

elif(select_Activity=="Model"):
    st.subheader("THE TRAINING DATASET")
    st.write("Training data will be solely used to train the model, and testing data will be used to test the accuracy of the model to find its effectiveness. The training data is consisting of _ variables that has been chosen to reduce misprediction of the model.")
    train_X
    train_y
    st.subheader("THE TESTING DATASET")
    st.write("testing dataset is the dataset that we feed into the finished model to enable us to quantify its effectiveness and accuracy.")
    test_X
    test_y
    select_Model=st.selectbox("Select Model",options=["Select Model","SVM","Decision Tree","Random Forest"])
    
    if select_Model=="SVM":
        st.subheader("SVM MODEL RESULTS:")
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(train_X,train_y)
        #Make prediction
        linear_pred = svclassifier.predict(test_X)
        #Evaluate SVM accuracy performance
        score2=accuracy_score(test_y,linear_pred)


        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(train_X, train_y)
        #Make prediction
        rbf_pred = svclassifier.predict(test_X)
        #Evaluate SVM accuracy performance
        score3=accuracy_score(test_y,rbf_pred)

        svclassifier = SVC(kernel='poly', degree=8)
        svclassifier.fit(train_X, train_y)
        poly_pred = svclassifier.predict(test_X)
        score4=accuracy_score(test_y,poly_pred)


        svclassifier = SVC(kernel='sigmoid')
        svclassifier.fit(train_X, train_y)
        sigmoid_pred = svclassifier.predict(test_X)
        score5=accuracy_score(test_y,sigmoid_pred)
        df = pd.DataFrame((["Linear",score2],["RBF",score3],["Poly",score4],["Sigmoid",score5]),columns=["Kernel","Accuracy Score"])
        st.table(df)
        
    elif select_Model=="Random Forest":
        rf10=RandomForestClassifier(n_estimators=10)
        rf10.fit(train_X,train_y)
        rf10Pred=rf10.predict(test_X)
        score6=accuracy_score(test_y,rf10Pred)

        rf25=RandomForestClassifier(n_estimators=25)
        rf25.fit(train_X,train_y)
        rf25Pred=rf25.predict(test_X)
        score7=accuracy_score(test_y,rf25Pred)

        rf50=RandomForestClassifier(n_estimators=50)
        rf50.fit(train_X,train_y)
        rf50Pred=rf50.predict(test_X)
        score8=accuracy_score(test_y,rf50Pred)

        rf100=RandomForestClassifier(n_estimators=100)
        rf100.fit(train_X,train_y)
        rf100Pred=rf100.predict(test_X)
        score9=accuracy_score(test_y,rf100Pred)
        df = pd.DataFrame((["10",score6],["25",score7],["50",score8],["100",score9]),columns=["number of estimators","Accuracy Score"])
        st.table(df)
        
    elif select_Model=="Decision Tree":
        car_pred_model= DecisionTreeClassifier()
        car_pred_model.fit(train_X,train_y)
        pred = car_pred_model.predict(test_X)
        score1=accuracy_score(test_y,pred)
        df = pd.DataFrame(([score1]),columns=["Accuracy Score"])
        st.table(df)

    st.header("CONCLUSION")
    st.write("In conclusion, the car recommendation system has successfully identified that Random Forest is the best machine learning model for making car recommendations to users based on their preferred features. Through a comparison of three models, namely support vector machine (SVM), decision tree and random forest, it was determined that the random forest model performed the best in terms of accuracy and overall performance. This model was able to accurately predict car recommendations based on the user's selected preferences, providing the most relevant results. The decision tree model was also found to be a close second, while the SVM model had the least performance. The comparison between the three models has helped to identify the best model to be used for this specific use case, and it may also be useful for other recommendation systems in the future.") 
    
    car_pred_model= DecisionTreeClassifier()
    car_pred_model.fit(train_X,train_y)
    pred = car_pred_model.predict(test_X)
    score1=accuracy_score(test_y,pred)
    
    rf100=RandomForestClassifier(n_estimators=100)
    rf100.fit(train_X,train_y)
    rf100Pred=rf100.predict(test_X)
    score9=accuracy_score(test_y,rf100Pred)
    
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(train_X,train_y)
    linear_pred = svclassifier.predict(test_X)
    score2=accuracy_score(test_y,linear_pred)    
    
    df = pd.DataFrame((["SVM",score2],["Random Forest",score9],["Decision Tree",score1]),columns=["Model","Score"])
    st.table(df)

elif(select_Activity=="Recommendation"):
    
    st.subheader("Car Recommendation based on features:")
    st.write("The user are required to choose from any of available features below, the system will display the rank of cars based on that preferences. The higher the rank, the accurate the car based on the user selected preferences.")
    EngineFuel=st.selectbox("Select fuel Preference", options=["Select Preference","diesel","electric","regular unleaded"])
    Engine_HP=st.selectbox("Select HP Preference", options=["Select Preference",200,170,210])
    Cylinder=st.selectbox("Select cylinder preference",options=["Select Preference",4,6,8])
    Transmission=st.selectbox("Select Transmission", options=["Select Preference", "MANUAL","AUTOMATIC"])
    DWheel=st.selectbox("Select Driven-Wheels", options=["Select Preference", "rear wheel drive","front wheel drive"])
    NoofDoor=st.selectbox("Select No of Door", options=["Select Preference", 2,4])
    Market=st.selectbox("Select Market Category", options=["Select Preference", "Luxury","Hatchback","Diesel","Performance"])
    Size=st.selectbox("Select car size", options=["Select Preference","Compact","Midsize","Large"])
    Style=st.selectbox("Select Car size", options=["Select Preference","Sedan","4dr SUV","Coupe", "Convertible"])
    Highway=st.selectbox("Select MPG on highway", options=["Select Preference",24,23,26,33])
    city=st.selectbox("Select city MPG", options=["Select preference", 17,16,15])
    
    hp=0
    Door=0
    h=0
    ct=0
    cy=0
    
    if(EngineFuel=="diesel"):
        fuel="diesel"
    elif(fuel=="electric"):
        fuel="electric"
    elif(fuel=="regular unleaded"):
        fuel="regular unleaded"
        
    if(Engine_HP==200):
        hp=200
    elif(Engine_HP==170):
        hp=170
    elif(Engine_HP==210):
        hp=210

    if(Cylinder==4):
        cy=4
    elif(Cylinder==6):
        cy=6
    elif(Cylinder==8):
        cy=8
    if(Transmission=="MANUAL"):
        T="MANUAL"
    elif(Transmission=="AUTOMATIC"):
        T="AUTOMATIC"
    if(DWheel=="rear wheel drive"):
        DW="rear wheel drive"
    elif(DWheel=="front wheel drive"):
        DW="front wheel drive"
    if(NoofDoor==2):
        Door=2
    elif(NoofDoor==4):
        Door=4
    if(Market=="Luxury"):
        Mkt="Luxury"
    elif(Market=="Hatchback"):
        Mkt="Hatchback"
    elif(Market=="Diesel"):
        Mkt="Diesel"
    elif(Market=="Performance"):
        Mkt="Performance"
    if(Size=="Compact"):
        size="Compact"
    elif(Size=="Midsize"):
        size="Midsize"
    elif(Size=="Large"):
        size="Large"
    if(Style=="Sedan"):
        style="Sedan"
    elif(Style=="4dr SUV"):
        style="4dr SUV"
    elif(Style=="Coupe"):
        style="Coupe"
    elif(Style=="Convertible"):
        style="Convertible"
    if(Highway==24):
        h=24
    elif(Highway==23):
        h=23
    elif(Highway==26):
        h=26
    elif(Highway==33):
        h=33
    if(city==17):
        ct=17
    elif(city==16):
        ct=16
    elif(city==15):
        ct=15

    if 'diesel' in carFuelLst:
        fuel=carFuelLst.index('diesel')#converting to num for predictor
    else:
        carFuelLst.append('diesel')
        fuel=carFuelLst.index('diesel')
        
    if 'regular unleaded' in carFuelLst:
        fuel=carFuelLst.index('regular unleaded')
    else:
        carFuelLst.append('regular unleaded')
        fuel=carFuelLst.index('regular unleaded')

    if 'electric' in carFuelLst:
        fuel=carFuelLst.index('electric')
    else:
        carFuelLst.append('electric')
        fuel=carFuelLst.index('electric')
        
    if 'MANUAL' in carTTLst:
        T=carTTLst.index('MANUAL')#converting to num for predictor
    else:
        carTTLst.append('MANUAL')
        Transmission=carTTLst.index('MANUAL')
        
    if 'AUTOMATIC' in carTTLst:
        T=carTTLst.index('AUTOMATIC')
    else:
        carTTLst.append('AUTOMATIC')
        T=carTTLst.index('AUTOMATIC')

    if 'rear wheel drive' in carDWLst:
        DW=carDWLst.index('rear wheel drive')
    else:
        carDWLst.append('rear wheel drive')
        DW=carDWLst.index('rear wheel drive')
        
    if 'front wheel drive' in carDWLst:
        DW=carDWLst.index('front wheel drive')#converting to num for predictor
    else:
        carDWLst.append('front wheel drive')
        DW=carDWLst.index('front wheel drive')
        
    if 'Luxury' in carCategory:
        Mkt=carCategory.index('Luxury')
    else:
        carCategory.append('Luxury')
        Mkt=carCategory.index('Luxury')

    if 'Hatchback' in carCategory:
        Mkt=carCategory.index('Hatchback')
    else:
        carCategory.append('Hatchback')
        Mkt=carCategory.index('Hatchback')
    if 'Diesel' in carCategory:

        Mkt=carCategory.index('Diesel')
    else:
        carCategory.append('Diesel')
        Mkt=carCategory.index('Diesel')

    if 'Performance' in carCategory:
        #print(carModel.index('Sunny'))
        Mkt=carCategory.index('Performance')
    else:
        carCategory.append('Performance')
        Mkt=carCategory.index('Performance')
        
    if 'Compact' in carVSLst:
        #print(carMakeLst.index('Nissan'))
        size=carVSLst.index('Compact')
    else:
        carVSLst.append('Compact')
        size=carVSLst.index('Compact')

    if 'Midsize' in carVSLst:
        #print(carModel.index('Sunny'))
        size=carVSLst.index('Midsize')
    else:
        carVSLst.append('Midsize')
        size=carVSLst.index('Midsize')
    
    if 'Large' in carVSLst:
        #print(carMakeLst.index('Nissan'))
        size=carVSLst.index('Large')
    else:
        carVSLst.append('Large')
        size=carVSLst.index('Large')

    if 'Sedan' in carVTLst:
        #print(carModel.index('Sunny'))
        style=carVTLst.index('Sedan')
    else:
        carVTLst.append('Sedan')
        style=carVTLst.index('Sedan')
    if '4dr SUV' in carMakeLst:
        #print(carMakeLst.index('Nissan'))
        style=carVTLstt.index('4dr SUV')
    else:
        carVTLst.append('4dr SUV')
        style=carVTLst.index('4dr SUV')

    if 'Coupe' in carVTLst:
        #print(carModel.index('Sunny'))
        style=carVTLst.index('Coupe')
    else:
        carVTLst.append('Coupe')
        style=carVTLst.index('Coupe')
    
    TestMake=['']
    if '' in carMakeLst:
        #print(carModel.index('Sunny'))
        TestMake=carMakeLst.index('')
    else:
        carMakeLst.append('')
        TestMake=carMakeLst.index('')
    
    d = {'MakeInd':[TestMake],'TypeInd':T,'DWInd':DW,'VSInd': size,'VTInd':style,'Engine HP':hp,'MktCategoryInd':Mkt,'FuelInd':fuel,'Number of Doors':Door,'highway MPG':h,'city mpg':ct,'Engine Cylinders':cy}
    df = pd.DataFrame(data=d)
    
    rf100=RandomForestClassifier(n_estimators=100)
    rf100.fit(train_X,train_y)
    rf100Pred=rf100.predict(test_X)
        
    CarPred = rf100.predict(df)
    st.subheader("Predicted Car: "+str(CarPred))