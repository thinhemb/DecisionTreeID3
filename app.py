# from random import sample
import streamlit as st
import pickle
from DecisionTree import*
from PIL import Image

if __name__=='__main__':

    st.title("*Nhận diện môt số loài hoa Iris*")

    ls=st.text_input("Chiều dài của đài hoa (cm) : ",placeholder=" 2.0 -> 4.4")
    ws=st.text_input("Chiều rộng của đài hoa (cm) : ",placeholder=" 4.0 -> 7.9")
    lp=st.text_input("Chiều dài của cánh hoa (cm) : ",placeholder=" 1.0 -> 6.9")
    wp=st.text_input("Chiều rộng của cánh hoa (cm) : ",placeholder=" 0.1 -> 2.6")
    path_model="./model.pkl"
    model = pickle.load(open(path_model, 'rb'))
    if st.button('Predicted') :
        sample={'sepal_length': float(ls), 'sepal_width': float(ws), 'petal_length': float(lp), 'petal_width': float(wp)}
        result=model.predict(sample)
        st.header("Predict: "+result)
        if result=="Iris-versicolor":
            img=Image.open("./image/Iris_versicolor_3.jpg")
        elif result=="Iris-virginica":
            img=Image.open("./image/Iris_virginica.jpg")
        else:
            img=Image.open("./image/Iris_setosa.jpg")
        cap="Hình ảnh loài hoa "+result
        st.image(img,caption=cap)
    
        
        
