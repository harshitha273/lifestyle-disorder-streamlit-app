import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Lifestyle Disorder Risk Analyzer & Diet Suggestion")

# Load the dataset
df = pd.read_csv("diet.csv")

# Select relevant nutritional features
features = ['DR1TKCAL', 'DR1TPROT', 'DR1TTFAT', 'DR1TCARB', 'DR1TSUGR']
df = df[features].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# Display sample data with cluster labels
st.subheader("Sample of Clustered Data")
st.write(df.head())

# Sidebar for user input
st.sidebar.header("Enter Your Nutrient Intake:")
user_input = []
for feature in features:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Predict cluster for user input
if any(user_input):
    user_scaled = scaler.transform([user_input])
    prediction = kmeans.predict(user_scaled)[0]

    st.sidebar.markdown(f"### 🧠 Predicted Risk Cluster: **{prediction}**")

    # Personalized Diet Advice
    st.subheader("🍽️ Suggested Diet Adjustments")

    if prediction == 0:
        st.markdown("🟥 **High Risk (Cluster 0):**")
        st.markdown("- High intake of calories, fats, and sugars detected.")
        st.markdown("- **Suggestions:**")
        st.markdown("  - Reduce processed and fried foods.")
        st.markdown("  - Cut down on sugary drinks and snacks.")
        st.markdown("  - Increase fiber and water intake.")
    elif prediction == 1:
        st.markdown("🟩 **Low Risk (Cluster 1):**")
        st.markdown("- Your nutrient intake appears healthy and balanced.")
        st.markdown("- **Suggestions:**")
        st.markdown("  - Maintain current dietary habits.")
        st.markdown("  - Continue regular physical activity and hydration.")
    elif prediction == 2:
        st.markdown("🟨 **Moderate Risk (Cluster 2):**")
        st.markdown("- Moderate levels of some nutrients detected.")
        st.markdown("- **Suggestions:**")
        st.markdown("  - Slightly reduce sugar and refined carbs.")
        st.markdown("  - Include more vegetables and lean proteins.")

else:
    st.warning("👉 Please enter all nutrient values in the sidebar to get your result.")
