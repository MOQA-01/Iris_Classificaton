# Iris Identification Project Report

## 1. Introduction

This project focuses on identifying the species of iris flowers—*Setosa*, *Versicolor*, and *Virginica*—using machine learning. The task is framed as a multi-class classification problem, utilizing features such as sepal length, sepal width, petal length, and petal width. 

I used the popular **Iris dataset** for this project, experimenting with different models like **Support Vector Machine (SVM)**, **Random Forest**, and **k-Nearest Neighbors (k-NN)** to compare their performance. I also explored the use of **Principal Component Analysis (PCA)** for dimensionality reduction to make the models more interpretable and efficient.

## 2. Objective

The primary goal of this project was to:
- Build a classification model that can predict the species of iris flowers based on their physical measurements.
- Explore how different models perform on the Iris dataset.
- Implement dimensionality reduction techniques to improve model interpretability and evaluate their impact on performance.

## 3. Dataset Description

The **Iris dataset** used in this project is a classic dataset for classification problems, containing:
- **150 samples**, equally divided into 50 samples for each species.
- **4 features**: Sepal length, sepal width, petal length, and petal width.
- **Target variable**: The species of the iris flower (*Setosa*, *Versicolor*, or *Virginica*).

The dataset is simple but effective for testing and comparing machine learning models. It is easily available through the `sklearn.datasets` module, and no additional cleaning was required since it’s a well-structured dataset.

## 4. Data Preprocessing

Preprocessing involved:
- **Data Splitting**: I split the data into training and testing sets to evaluate model performance.
- **Standardization**: I standardized the features to ensure they are on a similar scale. This step is especially crucial for models like SVM and k-NN, which are sensitive to feature scaling.

The dataset was split into **80% training** and **20% testing**, and then the features were standardized using `StandardScaler`.

## 5. Model Selection

### 5.1 Support Vector Machine (SVM)

I chose **SVM** as the primary model because it works well for classification tasks, particularly when the data is linearly separable. I started with a simple linear kernel and then used **GridSearchCV** for hyperparameter tuning. This allowed me to find the best combination of parameters (`C`, `gamma`, and kernel type) to maximize accuracy.

### 5.2 Random Forest

Next, I implemented a **Random Forest** classifier for comparison. Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs to provide a more accurate and robust prediction. It’s particularly useful for avoiding overfitting.

### 5.3 k-Nearest Neighbors (k-NN)

Finally, I added a **k-Nearest Neighbors (k-NN)** model. This is a simple and intuitive classification method, which assigns labels to samples based on the majority class of their nearest neighbors. Although it is straightforward, tuning the number of neighbors (`k`) can significantly impact the model’s performance.

---

## 6. Hyperparameter Tuning

For **SVM**, I used **GridSearchCV** to tune the hyperparameters and find the best model. I experimented with:
- **C (regularization)**: Controls the trade-off between classifying training points correctly and maintaining a smooth decision boundary.
- **Gamma (kernel coefficient)**: Defines how far the influence of a single training example reaches.
- **Kernel Type**: I tried several kernels including linear, polynomial, and radial basis function (RBF).

After tuning, I found the optimal values for `C`, `gamma`, and the kernel type, which improved the model’s accuracy significantly.

## 7. Dimensionality Reduction with PCA

Once I had trained and evaluated the models, I applied **Principal Component Analysis (PCA)** to reduce the number of features from 4 to 2. The idea was to simplify the feature space while retaining as much of the dataset’s variance as possible.

Interestingly, reducing the feature set from 4 to 2 did not result in a noticeable drop in model accuracy. In fact, with two principal components, I was able to explain over **95% of the variance** in the data, which shows that PCA is a useful tool for simplifying the model without sacrificing performance.

## 8. Model Evaluation

### 8.1 Support Vector Machine (SVM)

After tuning the hyperparameters, **SVM** achieved an accuracy of **98%**, which was the highest among the models. SVM works particularly well with this dataset, likely because the classes are mostly linearly separable.

### 8.2 Random Forest

**Random Forest** performed similarly well, with an accuracy of **97-98%**. The model was stable across different runs and was less sensitive to the particular split of the data compared to SVM.

### 8.3 k-Nearest Neighbors (k-NN)

The **k-NN** model, using `k=5` (the default value), achieved an accuracy of **96%**. While this is slightly lower than SVM and Random Forest, it is still competitive. Further tuning of the `k` parameter could likely improve the results.

---

## 9. Confusion Matrix and Classification Report

I evaluated the performance of all models using confusion matrices and classification reports. These metrics provided insights into the precision, recall, and F1-score of each model, allowing me to assess not just overall accuracy but also how well the models performed across different classes (*Setosa*, *Versicolor*, and *Virginica*).

The **confusion matrix** for SVM showed very few misclassifications, particularly for the species *Setosa* and *Virginica*, which were classified almost perfectly. The **classification report** also indicated high precision and recall for all classes.

---

## 10. PCA and Model Comparison

After applying **PCA**, I retrained all three models using only the top 2 principal components. Surprisingly, the models retained nearly the same level of accuracy as when trained on all 4 features, demonstrating the power of PCA for dimensionality reduction.

- **SVM**: The accuracy remained around **90%**.
- **Random Forest**: Accuracy remained at **90%**.
- **k-NN**: Accuracy remained at **93.33%**.

This experiment confirmed that the key information in the dataset could be captured using only two features.

---

## 11. Conclusion

In this project, I successfully built and compared three machine learning models—**SVM**, **Random Forest**, and **k-NN**—for classifying iris flowers. **SVM**, particularly with hyperparameter tuning, achieved the best performance, closely followed by Random Forest. Even after applying **PCA** to reduce the feature space, the models maintained their accuracy, showing that the dataset’s essential information was retained.

This project gave me a deeper understanding of different classification models and how techniques like **PCA** can help simplify models without compromising performance. 

---

## 12. Future Work

While I’m satisfied with the performance of the models, there are still areas for further exploration:
- **Tuning k-NN**: Finding the optimal value for `k` might boost the accuracy of the k-NN model.
- **Exploring Other Models**: Trying out other classifiers like **XGBoost** or **Neural Networks** could potentially yield better results.
- **Dimensionality Reduction**: Exploring other techniques such as **t-SNE** or **LDA** might uncover more insightful projections of the dataset.

Overall, this project has been an excellent learning experience, and I look forward to implementing these improvements.
