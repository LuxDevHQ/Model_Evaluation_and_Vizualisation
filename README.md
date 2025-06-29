#  Model Evaluation and Visualization  
## Topic: Evaluating Deep Learning Models  

##  Summary

In this lesson, we will:
- Understand how to **evaluate deep learning models** using core metrics like **accuracy**, **precision**, **recall**, **F1 score**, and the **confusion matrix**.
- Learn how to interpret **ROC curves** and **AUC scores**.
- Visualize training performance using **loss vs. accuracy curves**.
- Preview basic model interpretability techniques like **Grad-CAM** and **attention maps**.

---

## 1.  Why Model Evaluation Matters

> Training a model is only **half the battle** — knowing how well it performs in the real world is key.

---

###  Real-world Analogy

Imagine you're training a dog to detect drugs at the airport:
- **Accuracy**: How often it gets it right overall.
- **Precision**: Of all the times it barked "drugs!", how often was it correct?
- **Recall**: Of all the actual drug cases, how many did it catch?
- **F1 Score**: The balance between barking too much and missing drugs.

---

## 2.  Core Evaluation Metrics

### A. **Accuracy**
> Proportion of total predictions that were correct.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
````

### B. **Confusion Matrix**

> A table showing true vs. predicted values.

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

### C. **Precision**

> How many predicted positives are actually correct?
> Formula: `TP / (TP + FP)`

### D. **Recall (Sensitivity)**

> How many actual positives were detected?
> Formula: `TP / (TP + FN)`

### E. **F1 Score**

> Harmonic mean of precision and recall.

```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

---

###  Example – Evaluate Breast Cancer Model (from earlier)

```python
from sklearn.metrics import confusion_matrix, classification_report

# After model.predict()
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype('int')

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 3.  ROC Curve and AUC

### ROC (Receiver Operating Characteristic)

> Plots **True Positive Rate** vs **False Positive Rate** at various thresholds.

* Shows trade-off between **sensitivity** and **specificity**.
* Ideal model curves toward the top-left.

### AUC (Area Under the Curve)

* Ranges from 0.5 (random) to 1.0 (perfect).
* Measures the **overall ability** to distinguish classes.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')  # Random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

## 4.  Visualizing Training Performance

> Helps spot overfitting/underfitting early.

Use the `History` object from model training:

```python
history = model.fit(...)

import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

###  Real-World Analogy

Imagine you're coaching a student:

* If **train accuracy is high** but **val accuracy is low** → overfitting (they memorized answers).
* If both are low → underfitting (they didn’t learn enough).

---

## 5.  Model Interpretability (Preview)

We want to know **why** the model made a decision.

---

### A. Grad-CAM (for CNNs)

> Highlights which parts of an image influenced a decision.

* Used in medical imaging, surveillance, etc.
* Not usable with plain ANN (you’ll explore it in CNNs).

---

### B. Attention Maps (for NLP)

> Shows which **words** were "focused on" during text prediction.

* Used in machine translation, sentiment analysis, etc.
* You’ll explore it in upcoming NLP modules.

---

##  Summary Table

| Metric/Tool         | What it Does                         | Use Case Example                      |
| ------------------- | ------------------------------------ | ------------------------------------- |
| Accuracy            | Overall correct predictions          | Quick benchmark                       |
| Precision           | Measures false positives             | Email spam filter                     |
| Recall              | Measures false negatives             | Cancer detection                      |
| F1 Score            | Balance between precision and recall | Imbalanced classes                    |
| Confusion Matrix    | Detailed error breakdown             | Debugging binary classifiers          |
| ROC + AUC           | Class separation power               | Risk scoring, fraud detection         |
| Training Curves     | Visualize learning over time         | Model diagnosis                       |
| Grad-CAM, Attention | Explain decisions visually           | Interpretability in production models |

---

##  Final Thoughts

* Don’t rely on **just accuracy**, especially with **imbalanced data**.
* Use **confusion matrix**, **precision/recall**, and **ROC-AUC** for deeper insights.
* Always **visualize training** to spot issues early.
* Interpretation techniques like **Grad-CAM** and **attention maps** are vital for **trust and transparency**.

---

