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
# This line defines a new class called 'DiamondPricePredictionApp'. Classes are used to create objects and encapsulate data and functionality within those objects.

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
        
# This is the constructor method of the 'DiamondPricePredictionApp' class. It initializes the main window ('root') and sets its title. It then creates several buttons with different functionalities, such as loading the dataset, visualizing data, showing a heatmap, training a model, predicting a price, and quitting the application. Each button is associated with a corresponding method that will be executed when the button is clicked. The constructor also initializes several instance variables to store the dataset, feature matrices, target vectors, and the trained model.

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
                
# This method opens a file dialog to allow the user to select a CSV file containing the dataset. If a file is selected, it attempts to read the CSV file using Pandas and store the resulting DataFrame in the 'self.data' instance variable. If the file is successfully loaded, it displays a success message. If an exception occurs during the file loading process, it displays an error message with the exception details.

    def distribution_data(self):
        if self.data is not None:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data, x='price', kde=True)
            plt.title('Distribution of Diamond Price')
            plt.show()
        else:
            messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
            
# This method displays a histogram plot of the 'price' column in the dataset, representing the distribution of diamond prices. 
# It first checks if the 'self.data' instance variable is not None (i.e., a dataset has been loaded). If a dataset is available,
# it creates a new figure with a specified size, uses Seaborn's 'histplot' function to draw a histogram with a kernel density estimate (kde) overlay, 
# sets the plot title, and displays the plot. If no dataset has been loaded, it displays an error message.

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
            
# This method creates a grid of subplots to visualize the distribution of each feature in the dataset. It first checks if the 'self.data' instance variable is not None (i.e., a dataset has been loaded). 
# If a dataset is available, it creates a new figure with a specified size and calculates the number of rows needed to accommodate all features (excluding the index column). 
# It then iterates over each feature column, creating a distrubiton graph

    def show_heatmap(self):
            if self.data is not None:
                numerical_data = self.data.select_dtypes(include=['float64', 'int64']).drop('Unnamed: 0', axis=1)
                plt.figure(figsize=(10, 8))
                sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                plt.show()
            else:
                messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")

# This method displays a heatmap of the correlation between the numerical features in the dataset. It first checks if the 'self.data' instance variable is not None (i.e., a dataset has been loaded). 
# If a dataset is available, it selects only the numerical columns (float64 and int64) and drops the 'Unnamed: 0' column (which is the index column). 
# It then creates a new figure with a specified size, uses Seaborn's 'heatmap' function to draw a correlation heatmap with annotations and a specified color map, sets the plot title, 
# and displays the plot. If no dataset has been loaded, it displays an error message.

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

    # This method trains a machine learning model on the dataset. It first checks if the 'self.data' instance variable is not None (i.e., a dataset has been loaded). If a dataset is available, it performs several preprocessing steps:
    #   1. Drops any rows with missing values using 'dropna()'.
    #   2. Calculates the first and third quartiles (Q1 and Q3) and the interquartile range (IQR) of the 'price' column.
    #   3. Removes any rows where the 'price' value is outside the range of Q1 - 1.5*IQR and Q3 + 1.5*IQR (this removes outliers).
    #   4. Converts categorical features ('cut', 'color', and 'clarity') to one-hot encoded numerical features using 'get_dummies()'.
    #   5. Splits the dataset into features (X) and target (y) variables.
    #   6. Splits the data into training and testing sets using 'train_test_split()'.
    #
    # It then trains three different regression models (Linear Regression, Decision Tree Regression, and Random Forest Regression) using 3-fold cross-validation and selects the best model based on the highest mean R-squared score.
    #
    # The best model is then fitted on the training data, and a success message is displayed with the name of the best model. If no dataset has been loaded, it displays an error message.

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

# This method predicts the price of a new diamond based on the trained model. It first checks if the 'self.trained_model' instance variable is not None (i.e., a model has been trained). If a trained model is available, it creates a dictionary representing a new diamond with known features. It then converts this dictionary into a DataFrame and performs one-hot encoding on the categorical features to match the format of the training data. The resulting DataFrame is then reindexed to match the columns of the training data, with any missing columns filled with zeros.
#
# The 'predict()' method of the trained model is then called with the new diamond DataFrame, and the predicted price is obtained. A message box is displayed with the predicted price. If no model has been trained, it displays an error message.

if __name__ == "__main__":
    root = tk.Tk()
    app = DiamondPricePredictionApp(root)
    root.mainloop()

# It then creates an instance of the 'DiamondPricePredictionApp' class, passing the 'root' as an argument to the constructor. 
# Finally, it calls the 'mainloop()' method on the 'root' object, which starts the main event loop of the Tkinter application, 
# handling user interactions and updating the GUI as necessary.