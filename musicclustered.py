
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Amazon Music Clustering Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Amazon_Music_Clustered.csv")

df = load_data()

features = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"
]

page = st.sidebar.radio("Navigation",
["Home","Dataset","EDA","K-Means Analysis","PCA Visualization","Song Explorer"])

if page=="Home":
    st.title("🎵 Amazon Music Clustering Dashboard")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Songs",len(df))
    c2.metric("Artists",df["name_artists"].nunique())
    c3.metric("Clusters",df["Cluster"].nunique())
    c4.metric("Features",len(features))
    st.write("This dashboard explores Amazon Music songs clustered using K-Means.")

elif page=="Dataset":
    st.title("Dataset Explorer")
    q=st.text_input("Search song")
    temp=df.copy()
    if q:
        temp=temp[temp["name_song"].str.contains(q,case=False,na=False)]
    st.dataframe(temp,use_container_width=True)
    st.download_button("Download CSV",temp.to_csv(index=False),"Amazon_Music_Clustered.csv","text/csv")

elif page=="EDA":
    st.title("Exploratory Data Analysis")
    f=st.selectbox("Feature",features)
    fig,ax=plt.subplots()
    ax.hist(df[f],bins=30)
    ax.set_title(f"Distribution of {f}")
    st.pyplot(fig)

    fig,ax=plt.subplots()
    ax.boxplot(df[f],vert=False)
    ax.set_title(f"Boxplot of {f}")
    st.pyplot(fig)

    st.subheader("Correlation")
    fig=px.imshow(df[features].corr(),text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r")
    st.plotly_chart(fig,use_container_width=True)

elif page=="K-Means Analysis":
    st.title("Cluster Analysis")
    st.bar_chart(df["Cluster"].value_counts().sort_index())
    profile=df.groupby("Cluster")[features].mean().round(2)
    st.subheader("Cluster Profile")
    st.dataframe(profile,use_container_width=True)
    fig=px.imshow(profile,text_auto=True,aspect="auto",color_continuous_scale="YlGnBu")
    st.plotly_chart(fig,use_container_width=True)

elif page=="PCA Visualization":
    st.title("PCA Visualization")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
     ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Cluster"]
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=40,
    alpha=0.7,
    ax=ax
    )

    ax.set_title("PCA Cluster Visualization")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster")

    st.pyplot(fig)

elif page=="Song Explorer":
    st.title("Song Explorer")
    cl=st.selectbox("Cluster",sorted(df["Cluster"].unique()))
    out=df[df["Cluster"]==cl][["name_song","name_artists"]+features]
    st.dataframe(out,use_container_width=True)
