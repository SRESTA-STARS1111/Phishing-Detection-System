from tkinter import *
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score,recall_score, precision_score
from scipy.stats import randint, uniform
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SupportVectorClassifier

global filename
global df, X_train, X_test, y_train, y_test
global lgb_model

main = tk.Tk()
main.title("Phishing Detection System Through Hybrid Machine Learning Based on URL")
main.geometry("1600x1500")

import tkinter
from tkinter import PhotoImage
image_path=PhotoImage(file="hacker_glitch-wallpaper-1600x900.png")

bg_image=tkinter.Label(main,image=image_path)
bg_image.place(relheight=1,relwidth=1)

# Add a custom gradient background
def set_gradient_background(widget):
    gradient = PhotoImage(width=1600, height=150)
    for i in range(150):
        r = int(255 - (i * 255 / 150))
        g = int(105 - (i * 105 / 150))
        b = int(180 - (i * 180 / 150))
        color = f"#{r:02x}{g:02x}{b:02x}"
        gradient.put(color, to=(0, i, 1600, i+1))
    widget.create_image(0, 0, anchor=NW, image=gradient)
    widget.gradient = gradient  # Prevent garbage collection

# Create a canvas for the title background
title_canvas = Canvas(main, height=150, width=1600)
title_canvas.place(x=0, y=0)
set_gradient_background(title_canvas)

# Configure the title label with new font style and colors
title_font = ('Helvetica', 24, 'bold italic')
title = tk.Label(main, text='Phishing Detection System Through Hybrid Machine Learning Based on URL', font=title_font, bg='#ff69b4', fg='white')
title.place(relx=0.5, rely=0.1, anchor=CENTER)

font1 = ('times', 14, 'bold')
text = tk.Text(main, height=12, width=80, bg=main.cget("bg"), highlightthickness=0)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=160)
text.config(font=font1)


style = ttk.Style()
style.configure("TButton",
                font=('times', 13, 'bold'),
                background='blue',
                foreground='red',
                padding=10,
                focuscolor='none')
style.map("TButton",
          foreground=[('active', '#000000')],
          background=[('active', '#81C784')],
          relief=[('pressed', 'groove')],
          highlightcolor=[('focus', '#ffffff')],
          highlightbackground=[('focus', '#ffffff')])

def on_enter(e):
    e.widget['background'] = '#81C784'
    e.widget['foreground'] = '#000000'

def on_leave(e):
    e.widget['background'] = '#4CAF50'
    e.widget['foreground'] = '#ffffff'

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size(Number of Rows): " + str(len(df)) + "\n")
    text.insert(END, "Dataset Size(Number of Columns): " + str(len(df.columns)) + "\n")

def mapping():
    global df
    text.delete('1.0', END)
    text.insert(END, str(df['status'].value_counts()) + "\n")

    # Change status into int dtype with legitimate as 0 and phishing as 1
    mapping = {'legitimate': 0, 'phishing': 1}
    df['status'] = df['status'].map(mapping)
    
    text.insert(END, "\nAfter Mapping Of Dependent Label Completed\n")
    text.insert(END, str(df['status'].value_counts()) + "\n")

def Filter_columns():
    global df
    text.delete('1.0', END)
    text.insert(END, "Count Of Columns: " + str(len(df.columns)) + "\n\n\n")

    corr_matrix = df.corr()
    target_corr = corr_matrix['status']
    
    # Only choose features with absolute correlation value greater than 0.1
    threshold = 0.1
    relevant_features = target_corr[abs(target_corr) > threshold].index.tolist()

    text.insert(END, "AFTER CO-RELATION & FILTERING: " + "\n")


    text.insert(END, "Count of Relevant Features: " + str(len(relevant_features)) + "\n")


def Split_Data():
    global df, X_train, X_test, y_train, y_test
    
    # Only select relevant features
    corr_matrix = df.corr()
    target_corr = corr_matrix['status']
    threshold = 0.1
    relevant_features = target_corr[abs(target_corr) > threshold].index.tolist()
    X = df[relevant_features]
    X = X.drop('status', axis=1)
    y = df['status']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    text.delete('1.0', END)
    text.insert(END, "Shapes of X_train, X_test, y_train, y_test is: "+ "\n\n\n")

    text.insert(END, "Shape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n")


def Random_forest():
    global df, X_train, X_test, y_train, y_test,rf_accuracy,RF
    
    # Train Random Forest Classifier
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    
    # Make predictions
    y_pred = RF.predict(X_test)
    
    # Calculate accuracy score
    rf_accuracy = accuracy_score(y_test, y_pred)*100
    rf_recall = recall_score(y_test, y_pred)*100
    rf_f1 = f1_score(y_test, y_pred)*100
    fr_ps = precision_score(y_test, y_pred)*100

    
    # Display accuracy score
    text.delete('1.0', END)
    text.insert(END, f'RANDOM FOREST SCORES:\n\n\n\n')
    
    text.insert(END, f'Random Forest Accuracy Score Is: {rf_accuracy:.2f}\n')
    text.insert(END, f'Random Forest Recall Score Is: {rf_recall:.2f}\n')
    text.insert(END, f'Random Forest F1 Score Is: {rf_f1:.2f}\n')
    text.insert(END, f'Random Forest Precision Score Is: {fr_ps:.2f}\n')


def naive_Bayes():
    global df, X_train, X_test, y_train, y_test,nb_accuracy
    
    # Train Naive Bayes Classifier
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    
    # Make predictions
    y_pred = NB.predict(X_test)
    
    # Calculate scores
    nb_accuracy = accuracy_score(y_test, y_pred)*100
    nb_recall = recall_score(y_test, y_pred)*100
    nb_f1 = f1_score(y_test, y_pred)*100
    nb_ps = precision_score(y_test, y_pred)*100
    
    # Display scores
    text.delete('1.0', END)
    text.insert(END, f'NAIVE BAYES SCORES:\n\n\n\n')
    
    text.insert(END, f'Naive Bayes Accuracy Score Is: {nb_accuracy:.2f}\n')
    text.insert(END, f'Naive Bayes Recall Score Is: {nb_recall:.2f}\n')
    text.insert(END, f'Naive Bayes F1 Score Is: {nb_f1:.2f}\n')
    text.insert(END, f'Naive Bayes Precision Score Is: {nb_ps:.2f}\n')


def Gradient_boosting():
    global df, X_train, X_test, y_train, y_test,gb_accuracy
    
    # Train Gradient Boosting Classifier
    GB = GradientBoostingClassifier()
    GB.fit(X_train, y_train)
    
    # Make predictions
    y_pred = GB.predict(X_test)
    
    # Calculate scores
    gb_accuracy = accuracy_score(y_test, y_pred)*100
    gb_recall = recall_score(y_test, y_pred)*100
    gb_f1 = f1_score(y_test, y_pred)*100
    gb_ps = precision_score(y_test, y_pred)*100
    
    # Display scores
    text.delete('1.0', END)
    text.insert(END, f'GRADIENT BOOSTING SCORES:\n\n\n\n')
    
    text.insert(END, f'Gradient Boosting Accuracy Score Is: {gb_accuracy:.2f}\n')
    text.insert(END, f'Gradient Boosting Recall Score Is: {gb_recall:.2f}\n')
    text.insert(END, f'Gradient Boosting F1 Score Is: {gb_f1:.2f}\n')
    text.insert(END, f'Gradient Boosting Precision Score Is: {gb_ps:.2f}\n')


def KNN():
    global df, X_train, X_test, y_train, y_test,knn_accuracy
    
    # Train KNN Classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate scores
    knn_accuracy = accuracy_score(y_test, y_pred) * 100
    knn_recall = recall_score(y_test, y_pred) * 100
    knn_f1 = f1_score(y_test, y_pred) * 100
    knn_precision = precision_score(y_test, y_pred) * 100
    
    # Display scores
    text.delete('1.0', END)
    text.insert(END, 'K-NEAREST NEIGHBORS (KNN) SCORES:\n\n')
    text.insert(END, f'KNN Accuracy Score: {knn_accuracy:.2f}\n')
    text.insert(END, f'KNN Recall Score: {knn_recall:.2f}\n')
    text.insert(END, f'KNN F1 Score: {knn_f1:.2f}\n')
    text.insert(END, f'KNN Precision Score: {knn_precision:.2f}\n')



def SVC():
    global df, X_train, X_test, y_train, y_test,svc_accuracy
    
    # Train Support Vector Classifier
    svc = SupportVectorClassifier()
    svc.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svc.predict(X_test)
    
    # Calculate scores
    svc_accuracy = accuracy_score(y_test, y_pred) * 100
    svc_recall = recall_score(y_test, y_pred) * 100
    svc_f1 = f1_score(y_test, y_pred) * 100
    svc_precision = precision_score(y_test, y_pred) * 100
    
    # Display scores
    text.delete('1.0', END)
    text.insert(END, 'SUPPORT VECTOR CLASSIFIER (SVC) SCORES:\n\n')
    text.insert(END, f'SVC Accuracy Score: {svc_accuracy:.2f}\n')
    text.insert(END, f'SVC Recall Score: {svc_recall:.2f}\n')
    text.insert(END, f'SVC F1 Score: {svc_f1:.2f}\n')
    text.insert(END, f'SVC Precision Score: {svc_precision:.2f}\n')

def LR():
    global df, X_train, X_test, y_train, y_test,lr_accuracy
    
    # Train Logistic Regression Classifier
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate scores
    lr_accuracy = accuracy_score(y_test, y_pred) * 100
    lr_recall = recall_score(y_test, y_pred) * 100
    lr_f1 = f1_score(y_test, y_pred) * 100
    lr_precision = precision_score(y_test, y_pred) * 100
    
    # Display scores
    text.delete('1.0', END)
    text.insert(END, 'LOGISTIC REGRESSION SCORES:\n\n')
    text.insert(END, f'Logistic Regression Accuracy Score: {lr_accuracy:.2f}\n')
    text.insert(END, f'Logistic Regression Recall Score: {lr_recall:.2f}\n')
    text.insert(END, f'Logistic Regression F1 Score: {lr_f1:.2f}\n')
    text.insert(END, f'Logistic Regression Precision Score: {lr_precision:.2f}\n')


import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_chart():
    algorithms = ['Random Forest', 'Naive Bayes', 'Gradient Boosting', 'KNN', 'SVC', 'Logistic Regression']
    accuracies = [rf_accuracy, nb_accuracy, gb_accuracy, knn_accuracy, svc_accuracy, lr_accuracy]
    
    # Setting a seaborn style
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(algorithms, accuracies, color=sns.color_palette("viridis", len(algorithms)))
    
    plt.xlabel('Machine Learning Algorithms', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Scores (%)', fontsize=14, fontweight='bold')
    plt.title('Accuracy of Machine Learning Algorithms', fontsize=16, fontweight='bold')
    
    # Adding the data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2 - 0.15, yval + 1, f'{yval:.2f}%', color='black', fontweight='bold', fontsize=12)
    
    # Adding grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotating x-axis labels if they are too long
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Removing the top and right spines for a cleaner look
    sns.despine()
    
    plt.tight_layout()
    plt.show()



def predict():
    # Open file dialog to select a CSV file for prediction
    filename = filedialog.askopenfilename(initialdir="dataset", title="Select CSV File for Prediction", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    
    # Check if a file is selected
    if filename:
        # Read the data from the selected file
        data = pd.read_csv(filename)
        
        # Use the trained Random Forest model to make predictions
        predictions = RF.predict(data)



        
        # Display the predictions in the text box
        text.delete('1.0', END)
        text.insert(END, "Predictions:\n")
        for prediction in predictions:
            if prediction==1:
                text.insert(END, f'The Predicted Output Is PISHING'+'\n')
            else:
                text.insert(END, f'The Predicted Output Is LEGITIMATE'+'\n')
            
    else:
        messagebox.showinfo("Info", "No file selected for prediction.")



uploadButton = ttk.Button(main, text="Upload Dataset", command=upload, style="TButton", width=16)
uploadButton.place(x=50, y=500)
uploadButton.bind("<Enter>", on_enter)
uploadButton.bind("<Leave>", on_leave)

uploadButton = ttk.Label(main, text="File: ")
uploadButton.place(x=350, y=450)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=400, y=450)

Mapping_Dependent_Label = ttk.Button(main, text="Mapping_Dependent_Label", command=mapping, style="TButton",width=38)
Mapping_Dependent_Label.place(x=250, y=500)
Mapping_Dependent_Label.bind("<Enter>", on_enter)
Mapping_Dependent_Label.bind("<Leave>", on_leave)

Filter_columns = ttk.Button(main, text="Filter_columns", command=Filter_columns, style="TButton", width=16)
Filter_columns.place(x=650, y=500)
Filter_columns.bind("<Enter>", on_enter)
Filter_columns.bind("<Leave>", on_leave)

Split_Data = ttk.Button(main, text="Split_Data", command=Split_Data, style="TButton", width=16)
Split_Data.place(x=850, y=500)
Split_Data.bind("<Enter>", on_enter)
Split_Data.bind("<Leave>", on_leave)

Random_Forest = ttk.Button(main, text="Random_Forest", command=Random_forest, style="TButton", width=16)
Random_Forest.place(x=50, y=550)
Random_Forest.bind("<Enter>", on_enter)
Random_Forest.bind("<Leave>", on_leave)

naive_Bayas = ttk.Button(main, text="naive_Bayes", command=naive_Bayes, style="TButton", width=16)
naive_Bayas.place(x=250, y=550)
naive_Bayas.bind("<Enter>", on_enter)
naive_Bayas.bind("<Leave>", on_leave)


Gradiant_boosting = ttk.Button(main, text="Gradient_boosting", command=Gradient_boosting, style="TButton", width=16)
Gradiant_boosting.place(x=450, y=550)
Gradiant_boosting.bind("<Enter>", on_enter)
Gradiant_boosting.bind("<Leave>", on_leave)

KMM = ttk.Button(main, text="KNN", command=KNN, style="TButton", width=16)
KMM.place(x=650, y=550)
KMM.bind("<Enter>", on_enter)
KMM.bind("<Leave>", on_leave)

LinearSVC = ttk.Button(main, text="SVC", command=SVC, style="TButton", width=16)
LinearSVC.place(x=850, y=550)
LinearSVC.bind("<Enter>", on_enter)
LinearSVC.bind("<Leave>", on_leave)

Logistic_Regression = ttk.Button(main, text="Logistic_Regression", command=LR, style="TButton", width=18)
Logistic_Regression.place(x=1050, y=550)
Logistic_Regression.bind("<Enter>", on_enter)
Logistic_Regression.bind("<Leave>", on_leave)

plotButton = ttk.Button(main, text="Plot Results", command=plot_bar_chart, style="TButton", width=16)
plotButton.place(x=50, y=600)
plotButton.bind("<Enter>", on_enter)
plotButton.bind("<Leave>", on_leave)

predict_button = ttk.Button(main, text="Prediction", command=predict, style="TButton", width=16)
predict_button.place(x=250, y=600)
predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)

#main.config(bg='#32d1a7')
main.mainloop()
