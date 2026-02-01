# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 22:39:38 2026

@author: mkhab
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

st.title("Itumeleng Mkhabela – Research Profile")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "About Me", 
    "Education", 
    "Research & Field experience",
    "Publication",
    "Professional interets", 
    "My data analysis work", 
    "Contact"
])

with tab1:
    st.header("About Me")
    st.write("MSc Agronomy graduate with hands-on experience in field trials, seed quality, and lab evaluations. I am exploring data analysis and data science to generate insights and support data-driven decisions in research.")
    image = Image.open(r"C:\Users\mkhab\Downloads\python project 1\python_project1\Streamlit\streamlit_files\IMG-20231020-WA0047.jpg")
    
    image = image.resize((600, int(600 * image.height / image.width)))
    
    st.image(image)

with tab2:
    st.header("Educational Background")
    st.markdown("""
    - MSc Agronomy - University of Limpopo, 2026
    - BSc Plant Production - University of Limpopo, 2023
            """)

with tab3: 
    st.header("Research & Field Experience") 
    st.write("My research focused on hail damage in potato crops, aiming to identify resilient cultivars and provide solutions for sustainable production. This involved field trials, data collection on crop performance, and laboratory evaluations. The research contributes to improving crop management and supports data-driven decisions in agriculture. This experience also strengthened my interest in applying data analysis and visualization to enhance research outcomes and address practical challenges in agriculture.")


with tab4: 
    st.header("Publication") 
    st.markdown("""
    - https://sciforum.net/paper/view/28334
    """)

with tab5: 
    st.header("Professional interests") 
    st.markdown("""
    **Currently learning:** Python, SQL, Power BI, and Excel
    """
    )

with tab6:
    st.header("Correlation Analysis Demo")
    
    df = pd.read_csv("Hail simulation data.csv")
    
    #Full dataframe
    st.subheader("Full dataset")
    st.dataframe(df)

    # Exclude first 4 columns for correlation
    df_corr =df.iloc[:,4:]
    cols = df_corr.columns
    n = len(cols)

    corr_matrix = df_corr.corr()
    p_matrix = pd.DataFrame(np.zeros((n,n)), columns=cols, index=cols)

    for i in range(n): 
       for j in range(n): 
           r, p = pearsonr(df_corr.iloc[:, i], df_corr.iloc[:, j]) 
           p_matrix.iloc[i,j] = p

    # Create annotation matrix for heatmap
    annot_matrix = np.empty((n, n), dtype=object)
    for i in range(n): 
       for j in range(n): 
           if i > j:  # lower triangle: correlation coefficients
            annot_matrix[i,j] = f"{corr_matrix.iloc[i,j]:.2f}"
           elif i < j:  # upper triangle: p-values
            annot_matrix[i,j] = f"{p_matrix.iloc[i,j]:.2f}"
           else:  # diagonal
            annot_matrix[i,j] = "--"

    # Create mask for correlation colors (only lower triangle colored)
    mask_corr = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix, mask=mask_corr, annot=annot_matrix, fmt="", cmap="coolwarm",
                cbar=True, xticklabels=cols, yticklabels=cols, ax=ax, annot_kws={"size":10})
    
    # Overlay p-values with custom colors
    for i in range(n):
        for j in range(n):
            if i < j:  # upper triangle
                p_val = p_matrix.iloc[i,j]
                text_color = "black" if p_val < 0.05 else "white"
                rect_color = "white" if p_val < 0.05 else "black"
                ax.text(j+0.5, i+0.5, f"{p_val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10,
                        bbox=dict(facecolor=rect_color, edgecolor='none', boxstyle='round,pad=0.2'))
    
    
    
    # Show plot in Streamlit
    st.subheader("Correlation Matrix (lower=r, upper=p-values)")
    st.write("This correlation matrix shows the pairwise relationships between numeric variables in the potato hail recovery dataset. "
             "The lower triangle displays Pearson correlation coefficients (r), indicating the strength and direction of the relationships, "
             "while the upper triangle shows the corresponding p-values to assess statistical significance at the 0.05 level. "
             "Black blocks in the upper triangle highlight significant relationships (p < 0.05), and white blocks indicate non-significant relationships. "
             "This helps to understand how different traits influence the recovery of potato cultivars after hail damage.")
    st.pyplot(fig)
    
#contact information
with tab7:
    st.header("Contact Me")
    st.markdown(f"- **Email:** mkhabelaitumeleng483@gmail.com")
    st.markdown(f"- **Phone:** 065 544 6895")
