import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample extracted leads data (replace with your own CSV export)
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

# Features and label
X = df[['Company_Size', 'Intent_Keyword', 'Website_Visits', 'Prev_Engaged']]
y = df['Converted']

# Model training (split for validation on more data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)

# Predict on test/new leads
y_pred = clf.predict(X_test)
print("Test Lead Conversion Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# To score new leads:
new_leads = pd.DataFrame({
    'Company_Size': [100, 230],
    'Intent_Keyword': [1, 0],
    'Website_Visits': [12, 6],
    'Prev_Engaged': [0, 1]
})
scores = clf.predict_proba(new_leads)[:,1]      # Probability of conversion
print("Lead Conversion Scores:", scores)
