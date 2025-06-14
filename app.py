import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Title
st.title("Lifestyle Disorder Risk Clustering using Dietary Data")

# Load Dataset
st.header("1. Upload NHANES Diet Dataset")
uploaded_file = st.file_uploader("Upload diet.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Select relevant features
    st.subheader("2. Preprocessing & Feature Selection")
    features = ['DR1TKCAL', 'DR1TTFAT', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR']
    if all(col in df.columns for col in features):
        df_selected = df[features].dropna()
        st.write("Sample of Cleaned Data:", df_selected.head())

        # Clustering
        st.subheader("3. KMeans Clustering (k=3)")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_selected['Cluster'] = kmeans.fit_predict(df_selected)

        # Show cluster centroids
        st.write("📊 Cluster Centroids:")
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        st.write(centroids)

        # Visualize clusters
        st.subheader("4. Cluster Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_selected['DR1TKCAL'], df_selected['DR1TTFAT'], c=df_selected['Cluster'], cmap='viridis')
        ax.set_xlabel("Calories")
        ax.set_ylabel("Fat")
        ax.set_title("Clusters based on Calorie & Fat Intake")
        st.pyplot(fig)

        st.success("✅ Model completed successfully!")
    else:
        st.error("❌ Required columns not found in dataset.")
