import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("bookstore_visitors_synthetic.csv")

# Use only numeric features
X = df[['browsing_time_sec', 'num_books_bought']]

# Try with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans.fit_predict(X)

# Save the clustered data to a new CSV file
df.to_csv("bookstore_clusters.csv", index=False)

# Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['browsing_time_sec'], df['num_books_bought'], c=df['cluster'], cmap='viridis')
plt.title('K-Means Clustering of Bookstore Visitors (k=2)')
plt.xlabel('Browsing Time (sec)')
plt.ylabel('Number of Books Bought')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()



