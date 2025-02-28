# howMusicAffectsMentalHealth
# Music Effects on Mental Health - Machine Learning Analysis

## Overview
This project analyzes the effect of music on mental health using **Decision Tree** and **K-Nearest Neighbors (KNN)** classifiers. The dataset `music_effect_on_mental_health.csv` contains various features related to an individual's music preferences and their mental health conditions, and the goal is to predict how music affects mental health (Improve, noEffect, Worsen).

## Dataset
The dataset consists of 724 samples with the following features:
- **Age** (int)
- **primaryStreamingService** (categorical)
- **hoursPerDay** (float)
- **whileWorking** (categorical: Yes/No)
- **favGenre** (categorical: favorite music genre)
- **Anxiety, Depression, Insomnia, OCD** (float scores representing mental health conditions)
- **musicEffects** (target variable: Improve, noEffect, Worsen)

## Installation
To run this project, install the required dependencies:
```sh
pip install pandas numpy matplotlib scikit-learn pydotplus
```

## Data Preprocessing
1. **Load dataset:**
   ```python
   df = pd.read_csv("data/music_effect_on_mental_health.csv")
   ```
2. **Check dataset information:**
   ```python
   df.info()
   df.head()
   ```
3. **One-hot encoding for categorical features:**
   ```python
   X = pd.get_dummies(df.drop('musicEffects', axis=1))
   y = df['musicEffects']
   ```
4. **Split into train/test sets:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
   ```

## Model Training & Evaluation
### **Decision Tree Classifier**
1. **Find best max depth for decision tree:**
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.model_selection import cross_val_score
   for d in range(1, 12):
       model = DecisionTreeClassifier(max_depth=d)
       scores = cross_val_score(model, X_train, y_train, cv=5)
       print("Depth:", d, "Validation Accuracy:", scores.mean())
   ```
2. **Train the final decision tree model:**
   ```python
   model = DecisionTreeClassifier(max_depth=4)
   model.fit(X_train, y_train)
   print("Training Accuracy:", model.score(X_train, y_train))
   print("Test Accuracy:", model.score(X_test, y_test))
   ```
3. **Visualize decision tree:**
   ```python
   from sklearn.tree import export_graphviz
   from io import StringIO
   from IPython.display import Image
   import pydotplus
   
   dot_data = StringIO()
   export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=X.columns, class_names=["noEffect", "Improve", "Worsen"])
   graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
   graph.write_png('plots/DecTree.png')
   Image(graph.create_png())
   ```

### **K-Nearest Neighbors Classifier (KNN)**
1. **Feature scaling:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```
2. **Find best k value:**
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   for k in range(1, 30):
       clf = KNeighborsClassifier(n_neighbors=k)
       clf.fit(X_train_scaled, y_train)
       scores = cross_val_score(clf, X_train_scaled, y_train, cv=10)
       print("k:", k, "Validation Accuracy:", scores.mean())
   ```
3. **Train final KNN model:**
   ```python
   clf = KNeighborsClassifier(n_neighbors=26)
   clf.fit(X_train_scaled, y_train)
   print("Training Accuracy:", clf.score(X_train_scaled, y_train))
   print("Test Accuracy:", clf.score(X_test_scaled, y_test))
   ```

## Performance Evaluation
1. **Confusion Matrix:**
   ```python
   from sklearn.metrics import confusion_matrix
   yPredict = clf.predict(X_test_scaled)
   cm = confusion_matrix(y_test, yPredict)
   print(cm)
   ```
2. **Overfitting analysis:**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(training_accuracy, label="Training Accuracy")
   plt.plot(test_accuracy, label="Test Accuracy")
   plt.xlabel("n_neighbors")
   plt.ylabel("Accuracy")
   plt.legend()
   plt.title('Overfitting with Small value of k')
   plt.gca().invert_xaxis()
   plt.savefig('plots/KNN_overfitting.png')
   ```

## Visualizations
- **Scatter plots:**
   ```python
   import numpy as np
   d = np.array(df)
   plt.scatter(d[d[:,9] == "Improve", 5], d[d[:,9] == "Improve", 2], c='lightgreen', marker='s', label='Improve')
   plt.scatter(d[d[:,9] == "noEffect", 5], d[d[:,9] == "noEffect", 2], c='orange', marker='o', label='noEffect')
   plt.scatter(d[d[:,9] == "Worsen", 5], d[d[:,9] == "Worsen", 2], c='lightblue', marker='v', label='Worsen')
   plt.xlabel('Mental health')
   plt.ylabel('Streaming Services')
   plt.legend(loc="lower right")
   plt.savefig('plots/scatterDecTree.png')
   plt.show()
   ```

## Results
- **Decision Tree Accuracy:** ~74%
- **KNN Accuracy:** ~74.4% with k=26
- **Findings:**
  - Music streaming and favorite genre show correlation with mental health.
  - Overfitting observed with small k values in KNN.
  - Decision tree shows better explainability but lower validation accuracy.

## Conclusion
This project provides an **exploratory analysis** of how music affects mental health, leveraging **machine learning classification models**. The dataset, preprocessing, and model training help understand relationships between streaming services, music genres, and mental health conditions.

---
**Next Steps:**
- Feature engineering to improve model performance.
- Explore deep learning techniques.
- Deploy model using FastAPI.

### **Author:**
- [Your Name]
- [Your Email]
- [Your GitHub]

