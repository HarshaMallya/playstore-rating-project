import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("googleplaystore.csv")

# Drop corrupted row and filter ratings
df.drop(index=10472, inplace=True)
df = df[df['Rating'] <= 5]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Replace rare categories with 'OTHER'
cat_counts = df['Category'].value_counts()
rare_cats = cat_counts[cat_counts < 200].index
df['Category'] = df['Category'].apply(lambda x: 'OTHER' if x in rare_cats else x)

# Remove rows with inappropriate 'Content Rating'
df = df[~df['Content Rating'].isin(["Adults only 18+", "Unrated"])]

# Clean and convert 'Size'
def parse_size(val):
    try:
        if isinstance(val, str):
            if val.endswith('k'):
                return float(val[:-1]) / 1024
            elif val.endswith('M'):
                return float(val[:-1])
        return np.nan
    except:
        return np.nan

df['Size'] = df['Size'].replace('Varies with device', np.nan)
df['Size'] = df['Size'].apply(parse_size)
df['Size'].fillna(df['Size'].mean(), inplace=True)

# Clean and convert 'Installs'
df['Installs'] = df['Installs'].str.replace(",", "", regex=False).str.replace("+", "", regex=False)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Clean and convert 'Price'
df['Price'] = df['Price'].str.replace("$", "", regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Convert 'Reviews'
df['Reviews'] = df['Reviews'].replace('3.0M', '3000000.0')
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Convert 'Last Updated' and extract date features
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df['year_updated'] = df['Last Updated'].dt.year
df['month_updated'] = df['Last Updated'].dt.month

# Drop missing values
df.dropna(inplace=True)

# Label encode categorical columns for regression
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])
df['Type'] = le.fit_transform(df['Type'])
df['Content Rating'] = le.fit_transform(df['Content Rating'])
df['Genres'] = le.fit_transform(df['Genres'])

# EDA plots
plt.figure(figsize=(8, 6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
top_cats = df['Category'].value_counts().head(10)
sns.barplot(x=top_cats.index, y=top_cats.values)
plt.title('Top 10 Categories by App Count')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Type')
plt.title('Free vs Paid App Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Reviews', y='Rating', hue='Category', legend=False)
plt.title('Correlation Between Reviews and Ratings')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Content Rating', y='Rating')
plt.title('Content Rating vs App Rating')
plt.xticks(rotation=45)
plt.show()

# Top 10 most expensive apps
# Remove non-ASCII characters  from app names
df['App'] = df['App'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))

top_expensive_apps = df[['App', 'Price', 'Rating']].sort_values(by='Price', ascending=False).head(10)

print("Top 10 Most Expensive Apps:")
print(top_expensive_apps)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_expensive_apps, x='Price', y='App', palette='magma')
plt.title('Top 10 Most Expensive Apps')
plt.xlabel('Price ($)')
plt.ylabel('App')
plt.tight_layout()
plt.show()

# Prepare data for ML
X = df[['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Genres', 'Content Rating', 'year_updated', 'month_updated']]
y_class = df['Rating'].round().astype(int) - 1  # for classification
y_reg = df['Rating']  # for regression

# Normalize numerical features
scaler = MinMaxScaler()
X[['Reviews', 'Size', 'Installs', 'Price']] = scaler.fit_transform(X[['Reviews', 'Size', 'Installs', 'Price']])

# Classification models
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_c = sc.fit_transform(X_train_c)
X_test_c = sc.transform(X_test_c)

clf = XGBClassifier(n_estimators=300, learning_rate=0.05, n_jobs=4, verbosity=0)
clf.fit(X_train_c, y_train_c)
print("XGBoost Classification Accuracy:", clf.score(X_test_c, y_test_c))

# Regression model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_r, y_train_r)
y_pred_r = regressor.predict(X_test_r)
print("Random Forest Regression MSE:", mean_squared_error(y_test_r, y_pred_r))
print("Random Forest Regression R^2:", r2_score(y_test_r, y_pred_r))
