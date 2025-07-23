import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu
# x1 = np.linspace(-5, 5, 100)
# x2 = np.linspace(-5, 5, 100)

x1 = np.random.normal(0,1,1000)
x2 = np.random.normal(1,2,1000)

X1, X2 = np.meshgrid(x1, x2)


# Clip giá trị X2 vào [-3, 3]
X2_clipped = np.clip(X2, 0, 3)

noise = np.random.normal(0, 1, size=X1.shape)
Y = 3 * X1 + 20 * X2 + noise  # Y có shape (100, 100)

# Chuyển sang vector
x1_flat = X1.flatten()
x2_flat = X2.flatten()
y_flat = Y.flatten()

# --- Option 1: 2D scatter with color-coded Y ---
plt.figure(figsize=(6, 5))
plt.scatter(x1_flat, x2_flat, c=y_flat, cmap='coolwarm', s=10)
plt.colorbar(label="Y value")
plt.xlabel("X1")
plt.ylabel("X2 (clipped)")
plt.title("2D Scatter: Color represents Y")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Option 2: Collapse inputs using linear combination ---
x_combo = 3 * x1_flat + 2 * x2_flat

plt.figure(figsize=(6, 4))
plt.scatter(x_combo, y_flat, alpha=0.3, s=10)
plt.xlabel("3*X1 + 2*X2")
plt.ylabel("Y")
plt.title("Collapse X1 & X2 into single axis: (3X1 + 2X2) vs Y")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- HỒI QUY X1 -> Y ---
model_x1 = LinearRegression()
model_x1.fit(x1_flat.reshape(-1, 1), y_flat)
y_pred_x1 = model_x1.predict(x1_flat.reshape(-1, 1))

plt.scatter(x1_flat, y_flat, alpha=0.3, s=10, label="Data")
plt.plot(np.sort(x1_flat), y_pred_x1[np.argsort(x1_flat)], color='red', label="Regression Line")
plt.xlabel("X1")
plt.ylabel("Y")
plt.title(f"X1 vs Y with Regression Line - {model_x1.coef_}")
plt.legend()
plt.show()

# --- HỒI QUY X2 -> Y ---
model_x2 = LinearRegression()
model_x2.fit(x2_flat.reshape(-1, 1), y_flat)
y_pred_x2 = model_x2.predict(x2_flat.reshape(-1, 1))

plt.scatter(x2_flat, y_flat, alpha=0.3, s=10, color='orange', label="Data")
plt.plot(np.sort(x2_flat), y_pred_x2[np.argsort(x2_flat)], color='green', label="Regression Line")
plt.xlabel("X2")
plt.ylabel("Y")
plt.title(f"X2 vs Y with Regression Line {model_x2.coef_}")
plt.legend()
plt.show()
