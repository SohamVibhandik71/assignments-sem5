
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data[:, [2]]  # Use one feature
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
y_linear = linear.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_poly = model.predict(X_test_poly)

# Evaluate both models
for name, actual in [('Linear Regression', y_linear),
                     ('Polynomial Regression', y_poly)]:
    print("\n", name)
    print("MSE:", mean_squared_error(y_test, actual))
    print("MAE:", mean_absolute_error(y_test, actual))
    print("R2 Score:", r2_score(y_test, actual))
