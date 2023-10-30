import streamlit as st
from streamlit_modules import clustering
import time

st.title("Dominant Color Extraction:")
uploaded_file = st.file_uploader("Choose a JPEG image file")
if uploaded_file is not None:
    st.write("The uploaded image file")
    st.image(uploaded_file, width=400)
    st.write("")
    st.write("")
    st.write("")
    number = st.number_input('Insert an appropriate number for dominant color extraction and then click calculate button', min_value=1, step=1)
    st.write('The current number is ', number)
    if st.button('Calculate'):
        with st.spinner('Please Wait...'):
            res_graph = clustering.kmeans_clustering(uploaded_file, number)
            st.plotly_chart(res_graph, use_container_width=True)
        st.success('Done!')
        st.write("NOTE: Works well if the number is close to the actual number of dominant colors. if an image has 10 dominant colors by visually inspecting, then give a number around 10 but if given a number lets say 2 or a very high number the results may not make sense. Uses K-Means Clustering under the hood and the number you input is actually the number of clusters. For every cluster, mean of RGB is taken to arrive at the results.")