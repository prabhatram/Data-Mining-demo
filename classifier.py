import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("bookstore_behavior.csv")

#Select relevant features. These must be numeric or encoded. Encoded means categorical features converted to numbers. 
# Categorical means features that have discrete values like "red", "blue", "green" or "small", "medium", "large".
X = df[['browsing_time_sec', 'num_books_bought']]  # or any other numeric features
y = df['will_buy']  # target column (1 = will buy, 0 = won't)

#Split into train and test. You can use 70-30, 80-20, 90-10 splits. Here we use 80-20, because test_size = 0.2 (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)


# Unseen data point, for which we will predict the outcome
new_customer = pd.DataFrame({'browsing_time_sec': [402], 'num_books_bought': [7]})
predicted = model.predict(new_customer)[0]
print(f"Prediction for new customer: {predicted} (1=Will Buy, 0=Will Not Buy)")

#Visualize Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(model,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree: Predicting Book Purchase Behavior")
plt.show()
print("Accuracy:", model.score(X_test, y_test))

