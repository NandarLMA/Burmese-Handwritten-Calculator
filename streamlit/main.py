import streamlit as st
#import streamlit.elements.image as st_image
import cv2 as cv
import numpy as np
from streamlit_drawable_canvas import st_canvas
from image_segmentation import imageSegmentation, calculateResult
#from tensorflow.keras.models import load_model

#model = load_model('cnn_digits_model4.h5')
header = st.container()
dataset = st.container()
canvas = st.container()

with header:
  st.title("Burmese Handwritten Calculator မှ ကြိုဆိုပါတယ်!!")
  st.text("")

with dataset:
  st.subheader("ရည်ရွယ်ချက်")
  st.markdown("ကလေး၊ လူငယ်၊ လူကြီးမရွေး ကိန်းဂဏန်းဆိုင်ရာ တွက်ချက်မှုများတွင် အထောက်အကူပြုရန်နှင့်၊ တွက်ချက်မှုနှင့် ရင်းနှီးကျွမ်းဝင်မှု " 
  "မရှိသော လူများကို တစ်ဖက်တစ်လမ်းမှ ကူညီရန်။ ")
  st.text("")
  st.text("")

with canvas:
  st.subheader("သင်္ချာဂဏန်းများ ရေးသားရန်")
  st.text("")

  st.sidebar.header("Configuration")
  b_width = st.sidebar.slider("မင် အရွယ်အစားရွေးချယ်ရန် : ", 15, 50, 0)
  b_color = st.sidebar.color_picker("မင် အရောင်ရွေးချယ်ရန် : ")
  bg_color = st.sidebar.color_picker("နောက်ခံ အရောင်ရွေးချယ်ရန် : ", "#000")

  with st.form("my_form"):
    SIZE = 250
    canvas_result = st_canvas(
      stroke_width=b_width,
      stroke_color=b_color,
      background_color=bg_color,
      height=SIZE,
      width = 1800,
      drawing_mode="freedraw",
      key='canvas')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
      cv.imwrite(f"img.jpg",  canvas_result.image_data)
      img = canvas_result.image_data
      st.write('Model Input')
      st.image(img)

  if submitted:
    data_array = imageSegmentation(img)
    st.write(f'ခန့်မှန်း တန်ဖိုးများမှာ : {data_array}')
    result = calculateResult(data_array)
    print(result)
    st.write(f'ရလဒ်မှာ : {result}')
    #st.bar_chart()'''