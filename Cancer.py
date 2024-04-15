#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network classification
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_preds)
print("Neural Network Accuracy:", nn_accuracy)


# In[17]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Adaptive Decision Learner (ADL) model
class ADL:
    def __init__(self, base_classifier, n_iterations):
        self.base_classifier = base_classifier
        self.n_iterations = n_iterations
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            model = self.base_classifier()
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return np.mean(predictions, axis=1)

# Create and train the ADL model
adl = ADL(base_classifier=DecisionTreeClassifier, n_iterations=10)
adl.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred = adl.predict(X_test_scaled)

# Convert predicted probabilities to class labels
y_pred_binary = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("ADL Accuracy:", accuracy)


# In[18]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'seed': 42
}

# Train the XGBoost model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Predict on the testing set
y_pred_prob = bst.predict(dtest)
y_pred = np.round(y_pred_prob)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)


# In[49]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_rf = rf_clf.predict(X_test_scaled)

# Calculate accuracy for RandomForestClassifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForestClassifier Accuracy:", accuracy_rf)

# Define and train SVC
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_svm = svm_clf.predict(X_test_scaled)

# Calculate accuracy for SVC
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVC Accuracy:", accuracy_svm)

# Define and train LogisticRegression
lr_clf = LogisticRegression(solver='liblinear', random_state=42)
lr_clf.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_lr = lr_clf.predict(X_test_scaled)

# Calculate accuracy for LogisticRegression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("LogisticRegression Accuracy:", accuracy_lr)


# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define AdaBoost classifier
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model
adaboost_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], 1)),  # LSTM layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape input data for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test_reshaped, y_test)[1]
print("Accuracy:", accuracy)


# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])  # Adding an extra dimension for the time steps

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape features to be compatible with GRU
X = X.reshape(X.shape[0], 1, X.shape[1])  # Adding an extra dimension for the time steps

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # GRU layer with 64 units
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the testing set
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)


# In[7]:


import seaborn as sns
# Define colors used as colorcodes
blue = '#51C4D3' # To mark drinkable water
green = '#74C365' # To mark undrinkable water
red = '#CD6155' # For further markings
orange = '#DC7633' # For further markings

# Plot the colors as a palplot
sns.palplot([blue])
sns.palplot([green])
sns.palplot([red])
sns.palplot([orange])


# In[10]:


# Load your dataset from CSV
df = pd.read_csv('/Users/prithwishghosh/Downloads/data.csv')


# In[11]:


import matplotlib.pyplot as plt
# Clear matplotlib and set style 
plt.clf()
plt.style.use('ggplot')

# Create subplot and pie chart
fig1, ax1 = plt.subplots()
ax1.pie(df['diagnosis'].value_counts(), colors=[green, blue], labels=[ 'M', 'B'], autopct='%1.1f%%', startangle=0, rotatelabels=False)

#draw circle
centre_circle = plt.Circle((0,0),0.80, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  

# Set tighten layout and show plot 
plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt

# Data
ml_algorithms = ['XGboost', 'Neural Networks', 'CNN', 'RNN', 'AdaBoost', 
                 'Adaptive Decision Learner', 'LSTM', 'GRU', 'Random Forest', 
                 'SVM', 'Logistic Regression']
accuracy_scores = [95.6140, 97.368, 94.736, 94.7, 97.368, 93.859, 
                   94.736, 93.859, 96.4912, 98.2456, 97.368]

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(ml_algorithms, accuracy_scores, color='skyblue')
plt.xlabel('Machine Learning Algorithm', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy of Different Machine Learning Algorithms', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display values on top of bars
for i in range(len(ml_algorithms)):
    plt.text(i, accuracy_scores[i] + 0.3, f'{accuracy_scores[i]:.2f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

