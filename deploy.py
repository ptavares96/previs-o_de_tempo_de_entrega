import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# modelo
model_data = joblib.load('/home/paulo/Documents/delivery_projeto/best_trained_model.joblib')

model = model_data['best_model']
std_scaler = model_data['std_scaler']
min_max = model_data['min_max']


st.title('Previsão de tempo de entrega')
st.write('Adicione os detalhes da entrega:')

speed = st.number_input('Velocidade média em km/h')
distance = st.number_input('Distância (em km)')
Delivery_person_Ratings = st.number_input('Avaliação do entregador', min_value=0.0, max_value=5.0, step=0.1)
Delivery_person_Age = st.number_input('Idade do entregador', min_value=18.0, max_value=50.0, step=1.0)

order_types = {'Buffet': 'Buffet', 'Drinks': 'Bebidas', 'Meal': 'Refeição', 'Snack': 'Lanche'}
vehicle_types = {'bicycle': 'Bicicleta', 'electronic_scooter': 'Patinete elétrico', 'motorcycle': 'Motocicleta', 'scooter': 'Patinete'}


translated_order_types = [order_types[order] for order in order_types.keys()]
translated_vehicle_types = [vehicle_types[vehicle] for vehicle in vehicle_types.keys()]

selected_order_type = st.selectbox('Selecione o tipo de pedido', translated_order_types)
select_vehicle_type = st.selectbox('Selecione o tipo de veículo', translated_vehicle_types)

if st.button('Prever'):
    scaled_speed = std_scaler.transform(np.array([[speed]]))
    scaled_distance = std_scaler.transform(np.array([[distance]]))
    scaled_Delivery_person_Ratings = min_max.transform(np.array([[Delivery_person_Ratings]]))
    scaled_Delivery_person_Age = std_scaler.transform(np.array([[Delivery_person_Age]]))


    order_encoded = np.zeros(len(order_types))
    order_index = translated_order_types.index(selected_order_type)
    order_encoded[order_index] = 1

    vehicle_encoded = np.zeros(len(vehicle_types))
    vehicle_index = translated_vehicle_types.index(select_vehicle_type)
    vehicle_encoded[vehicle_index] = 1

    features = np.hstack((scaled_speed, scaled_distance, scaled_Delivery_person_Ratings, scaled_Delivery_person_Age, order_encoded.reshape(1,-1), vehicle_encoded.reshape(1,-1)))
    prediction = model.predict(features)
    st.write(f'Previsão de tempo de entrega: {prediction[0]:.0f} minutos')