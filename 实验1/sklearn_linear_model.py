import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X = np.arange(-5, 5, 0.3)[:32].reshape((32, 1))
y = -5 * X + 0.1 * np.random.normal(loc=0.0, scale=20.0, size=X.shape)
# print(X)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X[:25]
X_test = X[25:33]
y_train = y[:25]
y_test = y[25:33]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred))

plt.scatter(X, y, label='Samples')
plt.plot(X, -5 * X + 0.1, c='r', label='True function')
plt.plot(X, y_pred, c='b', label='Trained model')
plt.legend()
plt.show()
