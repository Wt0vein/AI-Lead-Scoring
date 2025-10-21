# AI-Lead-Scoring
This repo provides a lightweight, production-ready example for AI-powered lead scoring in a full stack SaaS workflow. It uses Python, pandas, and scikit-learn’s Random Forest to assign conversion probabilities to B2B marketing/sales leads based on intent and engagement features
Here’s a ready-to-use **GitHub README.md/demo** for your AI Lead Scoring module (plug-and-play for employer or interview challenge):

# Features

- **Automated lead scoring** using machine learning
- Trains on labeled lead data (convert/did not convert)
- Predicts probability to convert for new incoming leads
- Easily extensible (add features, new data sources, connect to API/UI)
- Jupyter compatible for rapid prototyping and demo

***

## Demo Notebook

### Requirements

- Python 3.7+
- `pandas`
- `scikit-learn`

Install dependencies:
```bash
pip install pandas scikit-learn
```

### Code Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load your historical lead data
data = {
    'Company': ['ABC Tech', 'XYZ Inc', 'DataCorp', 'BetaSoft'],
    'Industry': ['Software', 'Consulting', 'FinTech', 'Healthcare'],
    'Company_Size': [50, 200, 120, 75],
    'Intent_Keyword': [1, 0, 1, 0],    # 1 if recent intent/action detected (demo/book call), else 0
    'Website_Visits': [15, 5, 23, 8],
    'Prev_Engaged': [1, 0, 1, 0],      # Contact replied in past? 1=yes, 0=no
    'Converted': [1, 0, 1, 0]          # Target label: did they actually convert?
}
df = pd.DataFrame(data)

# 2. Train/Test Split
X = df[['Company_Size', 'Intent_Keyword', 'Website_Visits', 'Prev_Engaged']]
y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)

# 4. Scoring on validation/test data
y_pred = clf.predict(X_test)
print(f\"Prediction: {y_pred}\")
print(f\"Accuracy: {accuracy_score(y_test, y_pred):.2f}\")

# 5. Predict on new leads
new_leads = pd.DataFrame({
    'Company_Size': [100, 230],
    'Intent_Keyword': [1, 0],
    'Website_Visits': [12, 6],
    'Prev_Engaged': [0, 1]
})
scores = clf.predict_proba(new_leads)[:,1]      # Probability to convert
print(f\"Lead Conversion Scores: {scores}\")
```

***

## Example Output

```
Prediction: [0]
Accuracy: 1.00
Lead Conversion Scores: [0.76 0.52]
```

***

## How to Use

1. Replace the sample data with your exported leads.
2. Add more features (job title, recency, engagement, etc.) for better accuracy.
3. Embed or call this model from your app backend or data pipeline to assign live scores.

***

## License

MIT

***

## Author

Nikunj Tikadia

