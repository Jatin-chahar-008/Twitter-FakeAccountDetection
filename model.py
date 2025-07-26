import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# --- Global variables to hold state (as in original code) ---
filename = None
ErrorrateMeans = []
AccuracyMeans = []

# --- Data and Model Functions (Original Logic Preserved) ---

def browse_file():
    """Opens a file dialog to select a CSV file and stores its path."""
    global filename
    # Use root as parent instead of creating a new Tk instance
    filepath = filedialog.askopenfilename(
        parent=root,
        title="Select a CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if filepath:
        filename = filepath
        # Update a label to show which file is selected
        file_label.config(text=f"File: {filename.split('/')[-1]}")
        print(f"Selected file: {filename}")


def run_naive_bayes():
    """Trains and evaluates the Multinomial Naive Bayes classifier."""
    if not filename:
        messagebox.showwarning("No File Selected", "Please browse for a dataset file first.")
        return

    global AccuracyMeans, ErrorrateMeans
    df = pd.read_csv(filename)
    msk = np.random.rand(len(df)) < 0.7
    train, test = df[msk], df[~msk]

    features = train.values[:, 0:7]
    labels = train.values[:, 8].astype('int')
    
    test_data = test.values[:, 0:7]
    test_labels = test.values[:, 8].astype('int')

    model = MultinomialNB()
    model.fit(features, labels)
    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions) * 100
    precision = precision_score(test_labels, predictions) * 100
    recall = recall_score(test_labels, predictions) * 100
    error_rate = 100 - accuracy
    
    AccuracyMeans.append(accuracy)
    ErrorrateMeans.append(error_rate)

    print("\n--- 1. Multinomial Naive Bayes ---")
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, predictions)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%\n")

    plot_pie_chart("Naive Bayes Performance", accuracy, error_rate)


def run_linear_svc():
    """Trains and evaluates the Linear SVC classifier."""
    if not filename:
        messagebox.showwarning("No File Selected", "Please browse for a dataset file first.")
        return

    global AccuracyMeans, ErrorrateMeans
    df = pd.read_csv(filename)
    msk = np.random.rand(len(df)) < 0.7
    train, test = df[msk], df[~msk]

    features = train.values[:, 0:7]
    labels = train.values[:, 8].astype('int')

    test_data = test.values[:, 0:7]
    test_labels = test.values[:, 8].astype('int')

    # Add max_iter to prevent potential convergence warnings
    model = LinearSVC(max_iter=5000, dual=True)
    model.fit(features, labels)
    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions) * 100
    precision = precision_score(test_labels, predictions) * 100
    recall = recall_score(test_labels, predictions) * 100
    error_rate = 100 - accuracy

    AccuracyMeans.append(accuracy)
    ErrorrateMeans.append(error_rate)
    
    print("\n--- 2. Linear SVC ---")
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, predictions)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%\n")

    plot_pie_chart("Linear SVC Performance", accuracy, error_rate)


def run_knn():
    """Trains and evaluates the K-Nearest Neighbors classifier."""
    if not filename:
        messagebox.showwarning("No File Selected", "Please browse for a dataset file first.")
        return
        
    global AccuracyMeans, ErrorrateMeans
    df = pd.read_csv(filename)
    msk = np.random.rand(len(df)) < 0.7
    train, test = df[msk], df[~msk]

    features = train.values[:, 0:7]
    labels = train.values[:, 8].astype('int')
    
    test_data = test.values[:, 0:7]
    test_labels = test.values[:, 8].astype('int')

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(features, labels)
    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions) * 100
    precision = precision_score(test_labels, predictions) * 100
    recall = recall_score(test_labels, predictions) * 100
    error_rate = 100 - accuracy
    
    AccuracyMeans.append(accuracy)
    ErrorrateMeans.append(error_rate)

    print("\n--- 3. K-Nearest Neighbors ---")
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, predictions)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%\n")
    
    plot_pie_chart("KNN Performance", accuracy, error_rate)


def compare_models():
    """Plots a bar chart comparing the performance of the classifiers."""
    if not AccuracyMeans:
        messagebox.showwarning("No Models Run", "Please run at least one classifier before comparing.")
        return

    N = len(AccuracyMeans)
    ind = np.arange(N)
    width = 0.35
    
    # Use different colors for better visibility
    p1 = plt.bar(ind, AccuracyMeans, width, color="#F08E05") 
    p2 = plt.bar(ind, ErrorrateMeans, width, bottom=AccuracyMeans, color="#7045F2")

    plt.ylabel('Scores (%)')
    plt.title('Performance By Classifier')
    # Make sure labels match the number of runs
    plt.xticks(ind, ('Naive Bayes', 'Linear SVC', 'KNN')[:N])
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Accuracy', 'Error Rate'))
    plt.tight_layout()
    plt.show()


def plot_pie_chart(title, accuracy, error_rate):
    """Helper function to create a pie chart."""
    labels = ['Accuracy', 'Error Rate']
    sizes = [accuracy, error_rate]
    explode = (0.1, 0)
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90, colors=['#F08E05', '#7045F2'])
    ax.axis('equal')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Manual Input Window ---
def create_input_window():
    """Creates the window for manual data entry and prediction."""
    input_window = tk.Toplevel(root)
    input_window.title("Manual Input for Prediction")
    input_window.geometry("400x550")
    input_window.resizable(False, False)

    # --- Helper function to create the input DataFrame ---
    def get_input_dataframe(entries):
        # The original code dropped 'Rejected Friend Requests'. This logic is preserved.
        input_dict = OrderedDict()
        for key, entry in entries.items():
            input_dict[key] = [entry.get()]

        input_frame = pd.DataFrame(input_dict)
        return input_frame[[
            'UserID', 'No Of Abuse Report', 'No Of Friend Requests Thar Are Not Accepted',
            'No Of Friends', 'No Of Followers', 'No Of Likes To Unknown Account',
            'No Of Comments Per Day'
        ]]

    # --- Prediction Functions (re-training each time as per original logic) ---
    def run_manual_prediction(model_type, entries):
        if not filename:
            messagebox.showwarning("No File", "Cannot predict without a base dataset file.", parent=input_window)
            return
        try:
            input_frame = get_input_dataframe(entries)
            print("Manual Input Data:")
            print(input_frame.loc[0])

            df = pd.read_csv(filename)
            msk = np.random.rand(len(df)) < 0.7
            train = df[msk]

            features = train.values[:, 0:7]
            labels = train.values[:, 8].astype('int')

            if model_type == 'NB':
                model = MultinomialNB()
            elif model_type == 'SVC':
                # Increased max_iter to prevent ConvergenceWarning
                model = LinearSVC(max_iter=10000, dual=True)
            else:  # KNN
                model = KNeighborsClassifier(n_neighbors=3)

            model.fit(features, labels)
            prediction = model.predict(input_frame.values[:, 0:7])

            # *** FIXED PART ***
            # Determine the result text and the style to apply
            if prediction[0] == 1:
                result_text = "Fake Account"
                result_style = "Fake.TLabel"
            else:
                result_text = "Genuine Account"
                result_style = "Genuine.TLabel"
            
            # Apply the text and style to the label
            result_label.config(text=f"Prediction: {result_text}", style=result_style)

        except (ValueError, KeyError) as e:
            messagebox.showerror("Input Error", f"Please ensure all fields are filled with valid numbers.\nDetails: {e}", parent=input_window)

    # --- UI Setup for Input Window ---
    main_frame = ttk.Frame(input_window, padding="10")
    main_frame.pack(fill="both", expand=True)

    fields = {
        'UserID': 'UserID', 'No Of Abuse Report': 'No Of Abuse Report',
        'Rejected Friend Requests': 'Rejected Friend Requests',
        'No Of Friend Requests Thar Are Not Accepted': 'No Of Friend Requests Not Accepted',
        'No Of Friends': 'No Of Friends', 'No Of Followers': 'No Of Followers',
        'No Of Likes To Unknown Account': 'No Of Likes To Unknown Account',
        'No Of Comments Per Day': 'No Of Comments Per Day'
    }

    entries = {}
    # Create labels and entries in a loop for cleaner code
    for i, (key, label_text) in enumerate(fields.items()):
        ttk.Label(main_frame, text=f"Enter {label_text}:").grid(row=i, column=0, sticky="w", pady=(5, 0))
        entry = ttk.Entry(main_frame, width=40)
        entry.grid(row=i, column=1, sticky="ew", pady=(5, 0))
        entries[key] = entry

    main_frame.grid_columnconfigure(1, weight=1)

    # Label to show the prediction result - apply a default style
    result_label = ttk.Label(main_frame, text="Prediction: -", style="Default.TLabel")
    result_label.grid(row=len(fields), column=0, columnspan=2, pady=20)

    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

    ttk.Button(button_frame, text="Predict (Naive Bayes)", command=lambda: run_manual_prediction('NB', entries)).pack(fill='x', pady=2)
    ttk.Button(button_frame, text="Predict (Linear SVC)", command=lambda: run_manual_prediction('SVC', entries)).pack(fill='x', pady=2)
    ttk.Button(button_frame, text="Predict (KNN)", command=lambda: run_manual_prediction('KNN', entries)).pack(fill='x', pady=2)


# --- Main Application Window Setup ---
root = tk.Tk()
root.title("Twitter Fake Account Detector")
root.geometry("650x450")
root.resizable(False, False)
root.configure(bg='#f0f0f0')

# Use a style for a modern look
style = ttk.Style(root)
style.theme_use('clam')

style.configure("Default.TLabel", font=("Helvetica", 12, "bold"))
style.configure("Genuine.TLabel", foreground="green", font=("Helvetica", 12, "bold"))
style.configure("Fake.TLabel", foreground="red", font=("Helvetica", 12, "bold"))

# --- Main UI Widgets ---
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill="both", expand=True)

# Title Label
ttk.Label(main_frame, text="Fake Account Detector", font=("Helvetica", 24, "bold", "italic"),
          foreground="dark violet").pack(pady=(0, 20))

# Browse Button
ttk.Button(main_frame, text="Browse Dataset File", command=browse_file, style='Accent.TButton').pack(fill='x', ipady=5)
file_label = ttk.Label(main_frame, text="File: Not Selected", font=("Helvetica", 9), foreground="gray")
file_label.pack(pady=(5, 20))

# Separator
ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)

# Classifiers Frame
classifier_frame = ttk.Frame(main_frame)
classifier_frame.pack(pady=10, fill='x', expand=True)
classifier_frame.columnconfigure((0, 1, 2), weight=1)

ttk.Button(classifier_frame, text="Naive Bayes", command=run_naive_bayes).grid(row=0, column=0, sticky="ew", padx=5, ipady=5)
ttk.Button(classifier_frame, text="Linear SVC", command=run_linear_svc).grid(row=0, column=1, sticky="ew", padx=5, ipady=5)
ttk.Button(classifier_frame, text="KNN", command=run_knn).grid(row=0, column=2, sticky="ew", padx=5, ipady=5)

# Action Buttons Frame
action_frame = ttk.Frame(main_frame)
action_frame.pack(pady=20, fill='x', expand=True)
action_frame.columnconfigure(0, weight=1)

ttk.Button(action_frame, text="Give Manual Input", command=create_input_window).grid(row=0, column=0, sticky="ew", ipady=5, pady=5)
ttk.Button(action_frame, text="Compare All Models", command=compare_models, style='Accent.TButton').grid(row=1, column=0, sticky="ew", ipady=5, pady=5)

root.mainloop()