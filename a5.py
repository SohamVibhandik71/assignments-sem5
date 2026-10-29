
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = sns.load_dataset('titanic')

# Data cleaning
df.drop_duplicates(inplace=True)
df.drop(columns=['deck', 'embark_town', 'alive'], inplace=True)

# Handle missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encoding categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Statistical analysis
print("Statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Visualizations
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

sns.boxplot(data=df.select_dtypes(include='number'))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Feature scaling
X = df.drop('survived', axis=1)
y = df['survived']

X = pd.get_dummies(X, drop_first=True)
X = StandardScaler().fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)
