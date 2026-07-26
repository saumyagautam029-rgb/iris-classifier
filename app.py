import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Classifier")

data = load_iris(as_frame=True)
X = data.data
y = data.target

model = RandomForestClassifier()
model.fit(X, y)

sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width", 0.1, 2.5, 0.2)

pred = model.predict([[sl, sw, pl, pw]])
st.write("Flower:", data.target_names[pred[0]])