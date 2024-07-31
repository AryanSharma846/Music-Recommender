import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Specify the path to the zip file
zip_file_path = 'C:/Users/aweso/Downloads/arch.zip'  # Update this path as needed

# Open the zip file and read the CSV file
with zipfile.ZipFile(zip_file_path, 'r') as z:
    #print("Contents of the zip file:")
    #for file in z.namelist():
        #print(file)

    # After identifying the correct file name, use it here
    csv_file_name = 'completelyfixed.csv'  # Update this with the correct name from the list

    with z.open(csv_file_name) as f:
        data = pd.read_csv(f, encoding='latin1')  # Try 'latin1', 'iso-8859-1', or 'cp1252' if needed

# Display the column headers to verify the file content
#print(data.columns)

# Define the features to extract
numerical_features = ['instrumentalness_%', 'valence_%', 'danceability_%', 'energy_%', 
                      'acousticness_%', 'liveness_%', 'speechiness_%']
categorical_features = ['artist(s)_name', 'key', 'mode']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform the data
preprocessed_features = preprocessor.fit_transform(data[numerical_features + categorical_features])

# Display the shape of the preprocessed features
#print(preprocessed_features.shape)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(preprocessed_features)

# Add the cluster labels to the original data
data['cluster'] = clusters

# Function to recommend songs
def recommend_songs(song_id, data, num_recommendations=5):
    song_details = data.loc[song_id, ['track_name', 'artist(s)_name']].to_dict()
    song_cluster = data.loc[song_id, 'cluster']
    similar_songs = data[data['cluster'] == song_cluster].index.tolist()
    similar_songs.remove(song_id)
    recommendations = data.loc[similar_songs[:num_recommendations], ['track_name', 'artist(s)_name', 'cluster']]
    
    return song_details, recommendations

# Test the recommendation system
song_id = 526  # Example song ID
song_details, recommendations = recommend_songs(song_id, data)
print("Song details:")
print(f"Track Name: {song_details['track_name']}")
print(f"Artist(s) Name: {song_details['artist(s)_name']}")
print("\nRecommended songs:")
print(recommendations)
