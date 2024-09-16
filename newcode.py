import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

def display_introduction():
    st.title('Welcome to CO2 Emissions Analysis and Prediction')
    st.write('This Streamlit app explores the relationship between vehicle characteristics and CO2 emissions, providing insightful visualizations and predictive capabilities.By analyzing a dataset of vehicle specifications and CO2 emissions, we aim to understand how factors such as engine size, cylinders, and fuel consumption impact environmental footprint.You can navigate through different sections to explore various visualizations depicting CO2 emissions trends across different vehicle attributes.Additionally, you can utilize our predictive model to estimate CO2 emissions based on input parameters, helping to make informed decisions towards reducing carbon footprint.')
    st.write('We hope this tool serves as a valuable resource for understanding and mitigating the environmental impact of vehicular emissions.')
    st.write('Select "Visualization" from the sidebar to explore visualizations or "Model" to predict CO2 emissions.')
df = pd.read_csv('co2 Emissions.csv')

fuel_type_mapping = {"Z": "Premium Gasoline","X": "Regular Gasoline","D": "Diesel","E": "Ethanol(E85)","N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
y = df_new_model['CO2 Emissions(g/km)']
model = RandomForestRegressor().fit(X, y)

model.feature_names_in_ = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']

with st.sidebar:
    st.markdown("# CO2 Emissions by Vehicle")
    user_input = st.selectbox('Please select',('Introduction','Visualization','Model'))

if user_input == 'Introduction':
    display_introduction()

elif user_input == 'Visualization':

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # import matplotlib.pyplot as plt.........new for matplot
    # plt.rcParams['figure.dpi'] = 150.....new for matplot (replace it )

    st.title('CO2 Emissions by Vehicle')
    st.header("Data We collected from the source")
    st.write(df)

    st.subheader('Brands of Cars')
    df_brand = df['Make'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(15, 6))
    fig1 = sns.barplot(data=df_brand, x="Make", y="Count")
    plt.xticks(rotation=75)
    plt.title("All Car Companies and their Cars")
    plt.xlabel("Companies")
    plt.ylabel("Cars")
    plt.bar_label(fig1.containers[0], fontsize=7)
    st.pyplot()
    st.write(df_brand)

    st.subheader('Top 25 Models of Cars')
    df_model = df['Model'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig2 = sns.barplot(data=df_model[:25], x="Model", y="Count")
    plt.xticks(rotation=75)
    plt.title("Top 25 Car Models")
    plt.xlabel("Models")
    plt.ylabel("Cars")
    plt.bar_label(fig2.containers[0])
    st.pyplot()
    st.write(df_model)

    st.subheader('Vehicle Class')
    df_vehicle_class = df['Vehicle Class'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 5))
    fig3 = sns.barplot(data=df_vehicle_class, x="Vehicle Class", y="Count")
    plt.xticks(rotation=75)
    plt.title("All Vehicle Class")
    plt.xlabel("Vehicle Class")
    plt.ylabel("Cars")
    plt.bar_label(fig3.containers[0])
    st.pyplot()
    st.write(df_vehicle_class)

    st.subheader('Engine Sizes of Cars')
    df_engine_size = df['Engine Size(L)'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig4 = sns.barplot(data=df_engine_size, x="Engine Size(L)", y="Count")
    plt.xticks(rotation=90)
    plt.title("All Engine Sizes")
    plt.xlabel("Engine Size(L)")
    plt.ylabel("Cars")
    plt.bar_label(fig4.containers[0])
    st.pyplot()
    st.write(df_engine_size)

    st.subheader('Cylinders')
    df_cylinders = df['Cylinders'].value_counts().reset_index().rename(columns={'count':'Count'})
    plt.figure(figsize=(20, 6))
    fig5 = sns.barplot(data=df_cylinders, x="Cylinders", y="Count")
    plt.xticks(rotation=90)
    plt.title("All Cylinders")
    plt.xlabel("Cylinders")
    plt.ylabel("Cars")
    plt.bar_label(fig5.containers[0])
    st.pyplot()
    st.write(df_cylinders)

    transmission_mapping = { "A4": "Automatic", "A5": "Automatic", "A6": "Automatic", "A7": "Automatic", "A8": "Automatic", "A9": "Automatic", "A10": "Automatic", "AM5": "Automated Manual", "AM6": "Automated Manual", "AM7": "Automated Manual", "AM8": "Automated Manual", "AM9": "Automated Manual", "AS4": "Automatic with Select Shift", "AS5": "Automatic with Select Shift", "AS6": "Automatic with Select Shift", "AS7": "Automatic with Select Shift", "AS8": "Automatic with Select Shift", "AS9": "Automatic with Select Shift", "AS10": "Automatic with Select Shift", "AV": "Continuously Variable", "AV6": "Continuously Variable", "AV7": "Continuously Variable", "AV8": "Continuously Variable", "AV10": "Continuously Variable", "M5": "Manual", "M6": "Manual", "M7": "Manual"}
    df["Transmission"] = df["Transmission"].map(transmission_mapping)
    st.subheader('Transmission')
    df_transmission = df['Transmission'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig6 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_transmission, x="Transmission", y="Count")
    plt.title("All Transmissions")
    plt.xlabel("Transmissions")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig6)
    st.write(df_transmission)

    st.subheader('Fuel Type')
    df_fuel_type = df['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig7 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_fuel_type, x="Fuel Type", y="Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig7)
    st.text("We have only one data on natural gas. So we cannot predict anything using only one data. That's why we have to drop this row.")
    st.write(df_fuel_type)

    st.subheader('After removing Natural Gas data')
    df_ftype = df_natural['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig8 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_ftype, x="Fuel Type", y="Count")
    plt.title("All Fuel Types")
    plt.xlabel("Fuel Types")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])
    st.pyplot(fig8)
    st.write(df_ftype)

    st.header('Variation in CO2 emissions with different features')
    st.subheader('CO2 Emission with Brand ')
    df_co2_make = df.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig8 = plt.figure(figsize=(20, 5))
    sns.barplot(data=df_co2_make, x="Make", y="CO2 Emissions(g/km)")
    plt.xticks(rotation=90)
    plt.title("CO2 Emissions variation with Brand")
    plt.xlabel("Brands")
    plt.ylabel("CO2 Emissions(g/km)")
    plt.bar_label(plt.gca().containers[0], fontsize=8, fmt='%.1f')
    st.pyplot(fig8)

    def plot_bar(data, x_label, y_label, title):
        plt.figure(figsize=(23, 5))
        sns.barplot(data=data, x=x_label, y=y_label)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.bar_label(plt.gca().containers[0], fontsize=9)

    st.subheader('CO2 Emissions variation with Vehicle Class')
    df_co2_vehicle_class = df.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_vehicle_class, "Vehicle Class", "CO2 Emissions(g/km)", "CO2 Emissions variation with Vehicle Class")
    st.pyplot()

    st.subheader('CO2 Emission variation with Transmission')
    df_co2_transmission = df.groupby(['Transmission'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_transmission, "Transmission", "CO2 Emissions(g/km)", "CO2 Emission variation with Transmission")
    st.pyplot()

    
    st.subheader('CO2 Emissions variation with Fuel Type')
    df_co2_fuel_type = df.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    plot_bar(df_co2_fuel_type, "Fuel Type", "CO2 Emissions(g/km)", "CO2 Emissions variation with Fuel Type")
    st.pyplot()

    
    st.header("Box Plots")
    plt.figure(figsize=(20, 10))
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
    for i, feature in enumerate(features, start=1):
        plt.subplot(2, 2, i)
        plt.boxplot(df_new[feature])
        plt.title(feature)
    st.pyplot()

    
    st.text("As we can see there are some outliers present in our Dataset")
    st.subheader("After removing outliers")
    st.write("Before removing outliers we have", len(df), "data")
    st.write("After removing outliers we have", len(df_new_model), "data")

    
    st.subheader("Boxplot after removing outliers")
    plt.figure(figsize=(20, 10))
    for i, feature in enumerate(features, start=1):
        plt.subplot(2, 2, i)
        plt.boxplot(df_new_model[feature])
        plt.title(feature)
    st.pyplot()



else:
    
    X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']

    
    model = RandomForestRegressor().fit(X, y)

    
if user_input == 'Model':
    st.title('CO2 Emission Prediction')
    st.write('Please enter the details of the vehicle:')
    
    
    engine_size = st.number_input('Engine Size (L)', min_value=0.1, value=1.0, step=0.1)
    cylinders = st.number_input('Cylinders', min_value=1, value=4, step=1)
    fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)', min_value=0.1, value=5.0, step=0.1)

    
    if engine_size < 0 or cylinders < 0 or fuel_consumption < 0:
        st.error('Inputs must be non-negative.')
    else:
        
        X_new = np.array([[engine_size, cylinders, fuel_consumption]])
        
        
        X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
        y = df_new_model['CO2 Emissions(g/km)']
        model = RandomForestRegressor()
        model.fit(X, y)
        
        
        prediction = model.predict(X_new)
        
        st.write('Estimated CO2 Emissions (g/km):', prediction[0])

  
    st.header('CO2 Emissions Statistics')
    st.write('Average CO2 Emissions per kilometer:', df['CO2 Emissions(g/km)'].mean(), 'g/km')
    st.write('Lowest CO2 Emissions per kilometer:', df['CO2 Emissions(g/km)'].min(), 'g/km')
    st.write('Highest CO2 Emissions per kilometer:', df['CO2 Emissions(g/km)'].max(), 'g/km')


