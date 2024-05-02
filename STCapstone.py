import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import math

class DiamondPricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diamond Price Prediction App")
        
        # Buttons
        self.load_data_button = tk.Button(self.root, text="Load Dataset", command=self.load_dataset)
        self.load_data_button.pack(pady=10)
        
        self.visualize_data_button = tk.Button(self.root, text="Distribution Data", command=self.distribution_data)
        self.visualize_data_button.pack(pady=5)

        self.visualize_data_button = tk.Button(self.root, text="Visualize Data", command=self.visualize_data)
        self.visualize_data_button.pack(pady=5)
        
        self.show_heatmap_button = tk.Button(self.root, text="Show Heatmap", command=self.show_heatmap)
        self.show_heatmap_button.pack(pady=5)
        
        self.train_model_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=5)
        
        self.predict_price_button = tk.Button(self.root, text="Predict Price", command=self.predict_price)
        self.predict_price_button.pack(pady=5)
        
        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=5)
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_model = None
        
    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
    
    def distribution_data(self):
        if self.data is not None:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data, x='price', kde=True)
            plt.title('Distribution of Diamond Price')
            plt.show()
        else:
            messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")

    def visualize_data(self):
        if self.data is not None:
            fig = plt.figure(figsize=(20, 15))
            nrows = math.ceil((len(self.data.columns) - 1) / 3)
            for i, col in enumerate(self.data.columns[1:]):
                ax = fig.add_subplot(nrows, 3, i+1)
                if self.data[col].dtype == 'object':
                    sns.countplot(data=self.data, x=col, ax=ax)
                else:
                    self.data[col].hist(bins=20, ax=ax)
                ax.set_xlabel(col)
            fig.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
    
    def show_heatmap(self):
        if self.data is not None:
            numerical_data = self.data.select_dtypes(include=['float64', 'int64']).drop('Unnamed: 0', axis=1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.show()
        else:
            messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
    
    def train_model(self):
        if self.data is not None:
            # Preprocessing
            self.data = self.data.dropna()
            Q1 = self.data['price'].quantile(0.25)
            Q3 = self.data['price'].quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data['price'] < (Q1 - 1.5 * IQR)) | (self.data['price'] > (Q3 + 1.5 * IQR)))]
            self.data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'])
            X = self.data.drop(['Unnamed: 0', 'price'], axis=1)
            y = self.data['price']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training models
            models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
            model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']
            results = [cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2').mean() for model in models]
            best_model_index = results.index(max(results))
            self.trained_model = models[best_model_index].fit(self.X_train, self.y_train)
            messagebox.showinfo("Success", f"Model trained successfully! Best model: {model_names[best_model_index]}")
        else:
            messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
    
    def predict_price(self):
        if self.trained_model is not None:
            new_diamond = {
                'carat': 0.23,
                'cut': 'Ideal',
                'color': 'E',
                'clarity': 'VS2',
                'depth': 61.5,
                'table': 55.0,
                'x': 3.95,
                'y': 3.98,
                'z': 2.43
            }
            new_diamond_df = pd.DataFrame([new_diamond])
            new_diamond_df = pd.get_dummies(new_diamond_df).reindex(columns=self.X_train.columns, fill_value=0)
            predicted_price = self.trained_model.predict(new_diamond_df)
            messagebox.showinfo("Prediction", f"The predicted price of the diamond is: {predicted_price[0]}")
        else:
            messagebox.showerror("Error", "No model trained. Please train a model first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiamondPricePredictionApp(root)
    root.mainloop()