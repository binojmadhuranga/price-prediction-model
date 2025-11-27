# 💻 Laptop Price Prediction Model

A machine learning project that predicts laptop prices in euros based on various hardware specifications and features using regression algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a **Random Forest Regressor** model to predict laptop prices based on specifications like RAM, weight, processor type, GPU, operating system, and other features. The model achieves high accuracy through feature engineering, data preprocessing, and hyperparameter tuning.

## 📊 Dataset

The dataset (`laptop_price.csv`) contains laptop specifications with the following original columns:
- `laptop_ID`: Unique identifier
- `Company`: Manufacturer brand
- `Product`: Product name
- `TypeName`: Laptop category (Notebook, Gaming, Ultrabook, etc.)
- `Inches`: Screen size
- `ScreenResolution`: Display resolution and type
- `Cpu`: Processor information
- `Ram`: Memory capacity
- `Gpu`: Graphics card information
- `OpSys`: Operating system
- `Weight`: Laptop weight
- `Price_euros`: Target variable (price in euros)

## ✨ Features

### Engineered Features:
1. **Touchscreen**: Binary indicator (1 if touchscreen, 0 otherwise)
2. **IPS**: Binary indicator for IPS display technology
3. **cpu_name**: Categorized as Intel Core i3/i5/i7, AMD, or Other
4. **gpu_name**: Categorized as AMD, Intel, or Nvidia
5. **Company**: Consolidated into major brands + "Other" category
6. **OpSys**: Simplified operating system categories

### Final Feature Set (36 features after one-hot encoding):
- Numerical: `Ram`, `Weight`, `Touchscreen`, `Ips`
- Categorical (One-Hot Encoded):
  - Company: Acer, Apple, Asus, Dell, HP, Lenovo, MSI, Other, Toshiba
  - TypeName: 2 in 1 Convertible, Gaming, Netbook, Notebook, Ultrabook, Workstation
  - OpSys: Android, Chrome OS, Linux, Mac OS X, No OS, Windows 10, Windows 10 S, Windows 7, macOS
  - CPU: AMD, Intel Core i3, Intel Core i5, Intel Core i7, Other
  - GPU: AMD, Intel, Nvidia

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/binojmadhuranga/price-prediction-model.git
cd price-prediction-model
```

2. **Create and activate virtual environment:**
```bash
python -m venv env
# Windows
.\env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. **Install required packages:**
```bash
pip install numpy pandas scikit-learn
```

## 💡 Usage

### Training the Model

Run the Jupyter notebook `laptop_price.ipynb` to:
1. Load and explore the dataset
2. Perform data preprocessing and feature engineering
3. Train multiple regression models
4. Optimize hyperparameters using GridSearchCV
5. Save the trained model

### Making Predictions

```python
import pandas as pd
import pickle

# Load the saved model columns
with open("model_columns.pkl", "rb") as file:
    columns = pickle.load(file)

# Example: Predict price for a Dell Notebook with specific specs
input_data = pd.DataFrame([[
    8,      # Ram (GB)
    1.3,    # Weight (kg)
    1,      # Touchscreen (Yes)
    1,      # IPS Display (Yes)
    0, 0, 0, 1, 0, 0, 0, 0, 0,  # Company (Dell)
    0, 0, 0, 1, 0, 0,           # TypeName (Notebook)
    0, 0, 0, 0, 0, 1, 0, 0, 0,  # OpSys (Windows 10)
    0, 0, 1, 0, 0,              # CPU (Intel Core i5)
    0, 1, 0                      # GPU (Intel)
]], columns=columns)

prediction = best_model.predict(input_data)
print(f"Predicted laptop price: €{prediction[0]:.2f}")
```

## 📈 Model Performance

Four regression models were evaluated:

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline linear model |
| **Lasso Regression** | Linear model with L1 regularization |
| **Decision Tree Regressor** | Non-linear tree-based model |
| **Random Forest Regressor** | Ensemble of decision trees (Best performer) |

### Hyperparameter Tuning

The Random Forest model was optimized using GridSearchCV with:
- **n_estimators**: [10, 50, 100]
- **criterion**: ['squared_error', 'absolute_error', 'poisson']
- **Cross-validation**: 5-fold

The best model configuration is saved and ready for predictions.

## 📁 Project Structure

```
price-prediction-model/
├── .git/                      # Git version control
├── .gitignore                 # Git ignore file
├── .ipynb_checkpoints/        # Jupyter notebook checkpoints
├── laptop_price.ipynb         # Main Jupyter notebook
├── laptop_price.csv           # Dataset (not included in repo)
├── model_columns.pkl          # Saved model column names
├── predictor.pickle           # Saved trained model (optional)
├── README.md                  # Project documentation
├── Lib/                       # Python libraries (virtual env)
├── Scripts/                   # Python scripts (virtual env)
└── pyvenv.cfg                 # Virtual environment config
```

## 🛠️ Technologies Used

- **Python 3.x**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
  - Linear Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - GridSearchCV for hyperparameter tuning
- **Jupyter Notebook**: Interactive development environment

## 🔮 Future Improvements

- [ ] Add more feature engineering (e.g., price per GB RAM, price per performance metric)
- [ ] Implement cross-validation metrics visualization
- [ ] Create a web interface for easy predictions (Flask/Streamlit)
- [ ] Add model interpretability analysis (SHAP values, feature importance)
- [ ] Expand dataset with more recent laptop models
- [ ] Implement additional ensemble methods (XGBoost, LightGBM)
- [ ] Add data visualization dashboard
- [ ] Deploy model as a REST API

## 📝 Notes

- The model is trained on historical laptop data; prices may vary with market conditions
- One-hot encoding increases dimensionality but improves model performance for categorical variables
- The Random Forest model provides the best balance between accuracy and generalization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Binoj Madhuranga**
- GitHub: [@binojmadhuranga](https://github.com/binojmadhuranga)
- Repository: [price-prediction-model](https://github.com/binojmadhuranga/price-prediction-model)

---

⭐ If you find this project useful, please consider giving it a star!
