from random import sample
from DecisionTree import *
import streamlit as st
import pickle

if __name__=='__main__':

    st.title("*Nhận diện môt số loài hoa*")

    ls=st.text_input("Length of the sepal (in cm)",placeholder="2.0 -> 4.4")
    ws=st.text_input("Width  of the sepal (in cm)",placeholder="4.0 -> 7.9")
    lp=st.text_input("Length of the petal (in cm)",placeholder="1.0 -> 6.9")
    wp=st.text_input("Width  of the petal (in cm)",placeholder="0.1 -> 2.6")
    path_model="DecisionTreeID/model.pkl"
    model = pickle.load(open(path_model, 'rb'))
    if st.button('Predicted') :
        sample={'sepal_length': float(ls), 'sepal_width': float(ws), 'petal_length': float(lp), 'petal_width': float(wp)}
        result=model.predict(sample)
        st.header("Predict: "+result)
    
        
        