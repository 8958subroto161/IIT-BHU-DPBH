import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import Entry, Label, Button, Text, Scrollbar

# Function to scrape text from a URL
def scrape_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text

# Function to load the database
def load_database(database_url):
    response = requests.get(database_url)
    lines = response.text.split('\n')
    data = [line.split('\t') for line in lines]
    return data

# Function to train the machine learning model
def train_model(data, text_column_index):
    texts = [row[text_column_index] if len(row) > text_column_index else '' for row in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Function to find the closest match in the database
def find_match(user_text, model, vectorizer, data):
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, model)
    closest_match_index = similarities.argmax()
    return data[closest_match_index]

# Function to find the corresponding "Pattern Category" for the input text
def find_pattern_category(input_text, model, vectorizer, data, text_column_index, category_column_index):
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, model)
    closest_match_index = similarities.argmax()
    matched_pattern = data[closest_match_index]
    return matched_pattern[category_column_index] if len(matched_pattern) > category_column_index else None

# Tkinter GUI
def run_program(database_url):
    user_url = url_entry.get()
    user_text = scrape_text(user_url)
    data = load_database(database_url)
    model, vectorizer = train_model(data, text_column_index=0)  # Assuming text is at index 0
    matched_pattern = find_match(user_text, model, vectorizer, data)

    result_text.config(state="normal")
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Matched pattern:\n")
    result_text.insert(tk.END, str(matched_pattern))
    result_text.insert(tk.END, "\nPattern Categories:\n")

    model, vectorizer = train_model(data, text_column_index=1)  # Assuming text is at index 1
    for t in range(len(matched_pattern)):
        if (t<2):
            input_text = matched_pattern[t]  # Assuming text is at index 1 in matched pattern
            pattern_category = find_pattern_category(input_text, model, vectorizer, data, text_column_index=1, category_column_index=3)
            result_text.insert(tk.END, f"{pattern_category}\n")

    result_text.config(state="disabled")

# Main Tkinter window
window = tk.Tk()
window.title("DARK PATTERN DETECTOR")
window.geometry("800x600")
window.configure(bg="#E8E8E8")

# Entry
url_entry_label = Label(window, text="Enter the URL to scrape:", bg="#E8E8E8")
url_entry_label.pack(pady=10)

url_entry = Entry(window, width=60)
url_entry.pack(pady=10)

# Button
run_button = Button(window, text="Search Patterns", command=lambda: run_program('https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv'), bg="#4CAF50", fg="white")
run_button.pack(pady=10)

# Result Text
result_text_label = Label(window, text="Results:", bg="#E8E8E8")
result_text_label.pack(pady=10)

result_text = Text(window, height=20, width=80, wrap="word")
result_text.pack()

scrollbar = Scrollbar(window, command=result_text.yview)
scrollbar.pack(side="right", fill="y")
result_text.config(yscrollcommand=scrollbar.set)

window.mainloop()
