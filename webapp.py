# webapp.py

from PIL import Image 
import os
import pandas as pd
import streamlit as st
import numpy as np
from main import main
from collections import Counter

path = os.path.dirname(__file__)

def app():
    """
    Streamlit page to showcase the implementation
    """
    st.title("Circuit Component Detector")
    menu = []
    menu.append('Upload image')
    for i in range(20):
        menu.append("Sample circuit "+str(i+1))
    choice = st.sidebar.radio("Sample Images", menu)
    
    if choice == "Upload image":
        image_file = st.file_uploader("Upload Image", type=['png','jpeg','jpg'])
    else:
        no = choice.split()
        image_file = 'Test_set/' + no[2] + '.PNG'

    if image_file is not None:
        st.header("Different stages in recognition of the circuit")
        
        img = Image.open(image_file)
        inp = np.array(img)
        rebuilt, comp, nodes, comp_list, jns_list, conn_list, det_texts = main(inp)

        st.image(img, width=480, caption='Scanned circuit')
        st.image(comp, width=480, caption='Detected components')
        st.image(nodes, width=480, caption='Nodes and terminals')
        st.image(rebuilt, width=480, caption='Rebuilt circuit')

        st.header("Description of the circuit")
        st.subheader("Components in the circuit are: ")

        # Create a table to display the separated components and values
        table_data_comp = {"Component Type": [], "Label": [], "Value": []}
        # Counter to count occurrences of each component
        component_counter = Counter()

        for i, _ in enumerate(comp_list):
            # Split the string based on space
            components = comp_list[i].split(' ', 2)

            # Add the components to the table_data_comp dictionary
            table_data_comp["Component Type"].append(components[0])
            table_data_comp["Label"].append(components[1])
            
            table_data_comp["Value"].append(det_texts[i] if i < len(det_texts) else "error")

            # Update component counter
            component_counter[components[0]] += 1

        # Create a DataFrame from the table_data_comp
        df_comp = pd.DataFrame(table_data_comp)

        # Display the table for components using Streamlit
        st.table(df_comp)

        # New table for component counts
        st.subheader("Component Counts")
        table_data_counts = {"Component Type": [], "Count": []}

        # Populate the table_data_counts with component counts
        for component, count in component_counter.items():
            table_data_counts["Component Type"].append(component)
            table_data_counts["Count"].append(count)

        # Create a DataFrame from the table_data_counts
        df_counts = pd.DataFrame(table_data_counts)

        # Display the table for component counts using Streamlit
        st.table(df_counts)

        st.subheader("Nodes in the circuit are: ")
        for i, _ in enumerate(jns_list):
            st.write(jns_list[i])
        st.subheader("Connections in the circuit are: ")
        for i, _ in enumerate(conn_list):
            st.write(conn_list[i])

if __name__ == '__main__':
    app()
