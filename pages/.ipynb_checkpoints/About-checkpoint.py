import streamlit as st
import pandas as pd
import pydeck as pdk


df_rest = pd.read_pickle("./cache/saved_restaurants.pkl")



st.markdown('# Easy Menu :burrito: :curry:')
st.markdown('## :wave: your ordering assistant :wave:')
st.markdown("## &emsp; _—learn more about a dish with pictures, wordclouds and reviews_")
st.markdown('#')
st.markdown('#')
st.markdown("### Developed By: &emsp;Chenhao Wu   ")
st.markdown("### Github:   &emsp; :blue[github.com/chenha0w/Easy_Menu]")
st.markdown("### Contact:  &emsp; :blue[chenhao.wu@mg.thedataincubator.com]")

st.info("Currently app only support searching restaurants shown below. Searching restaurants outside of the list may exceed memory limit.")

layer = pdk.Layer(
    "ScatterplotLayer",
    df_rest[['name','latitude','longitude']].dropna(),
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position='[longitude, latitude]',
    get_fill_color=[255, 140, 0],
    get_line_color=[0, 0, 0],
    get_radius=100
)
view_state = pdk.ViewState(latitude=32.911013, longitude=-117.14746, zoom=11, bearing=0, pitch=0)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"}))
#st.map(data=df_rest.dropna(subset=['latitude','longitude']))