import math

import streamlit as st
from demo.src.models import load_trained_model
from demo.src.utils import render_predict_from_pose, predict_to_image
from demo.src.config import MODEL_DIR, MODEL_NAME


@st.cache(show_spinner=False, allow_output_mutation=True)
def fetch_model():
    model, state = load_trained_model(MODEL_DIR, MODEL_NAME)
    return model, state


model, state = fetch_model()
pi = math.pi
st.set_page_config(page_title="DietNeRF Demo")
st.sidebar.header('SELECT YOUR VIEW DIRECTION')
theta = st.sidebar.slider("Theta", min_value=0., max_value=2.*pi,
                          step=0.5, value=0.)
phi = st.sidebar.slider("Phi", min_value=0., max_value=0.5*pi,
                        step=0.1, value=1.)
radius = st.sidebar.slider("Radius", min_value=2., max_value=6.,
                           step=1., value=3.)


pred_color, _ = render_predict_from_pose(state, theta, phi, radius)
im = predict_to_image(pred_color)

st.image(im, use_column_width=False)
