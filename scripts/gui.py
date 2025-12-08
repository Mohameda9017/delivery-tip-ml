import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os


# Load model + scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "..", "models", "final_xgboost.pkl"))
scaler = joblib.load(os.path.join(script_dir, "..", "models", "final_scaler.pkl"))


# Initialize tkinter GUI and init user input variables with default values
root = tk.Tk()
root.title("Tip Predictor")
distance_var = tk.StringVar(value="3.0")
subtotal_var = tk.StringVar(value="25.00")
wait_var = tk.StringVar(value="10")
weather_var = tk.StringVar(value="clear")
time_var = tk.StringVar(value="afternoon")
day_var = tk.StringVar(value="Fri")
rating_var = tk.StringVar(value="4")
items_var = tk.StringVar(value="3")
messages_var = tk.StringVar(value="1")
result_var = tk.StringVar(value="")


# Function to run when the "predict tip" button is clicked.
# Calculates tip and sets the corresponding tkinter UI box to display the result
def predict_tip():
    # Grab user input and assign variables
    distance = float(distance_var.get())
    subtotal = float(subtotal_var.get())
    wait_time = float(wait_var.get())
    weather = weather_var.get()
    time_of_day = time_var.get()
    day = day_var.get()
    comm_rating = int(rating_var.get())
    items = int(items_var.get())
    messages = int(messages_var.get())
   
    # Format data into the features the model expects
    data = {
        "distance_miles": [distance],
        "order_subtotal": [subtotal],
        "wait_time_minutes": [wait_time],
        "communication_rating": [comm_rating],
        "item_count": [items],
        "messages_sent": [messages],
        "weather_rain": [1 if weather == "rain" else 0],
        "weather_snow": [1 if weather == "snow" else 0],
        "time_of_day_morning": [1 if time_of_day == "morning" else 0],
        "time_of_day_night": [1 if time_of_day == "night" else 0],
        "day_of_week_Mon": [1 if day == "Mon" else 0],
        "day_of_week_Sat": [1 if day == "Sat" else 0],
        "day_of_week_Sun": [1 if day == "Sun" else 0],
        "day_of_week_Thu": [1 if day == "Thu" else 0],
        "day_of_week_Tue": [1 if day == "Tue" else 0],
        "day_of_week_Wed": [1 if day == "Wed" else 0],
    }
   
    # Transform data into a dataframe, feed it to the scaler, then feed the scaled data into the model to obtain the prediction
    # Divide the prediction by 100 and multiply by subtotal to get the tip amount
    df = pd.DataFrame(data)
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)[0]
    tip_amount = (prediction / 100) * subtotal
    result_var.set(f"Tip: {prediction:.1f}% (${tip_amount:.2f})")




# Display the GUI
row = 0
tk.Label(root, text="Distance (miles):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=distance_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Order Subtotal ($):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=subtotal_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Wait Time (min):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=wait_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Weather:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
ttk.Combobox(root, textvariable=weather_var, values=["clear", "rain", "snow"], state="readonly").grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Time of Day:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
ttk.Combobox(root, textvariable=time_var, values=["morning", "afternoon", "night"], state="readonly").grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Day of Week:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
ttk.Combobox(root, textvariable=day_var, values=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], state="readonly").grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Communication (1-5):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=rating_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Item Count:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=items_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Label(root, text="Messages Sent:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
tk.Entry(root, textvariable=messages_var).grid(row=row, column=1, padx=5, pady=2)
row += 1


tk.Button(root, text="Predict Tip", command=predict_tip).grid(row=row, column=0, columnspan=2, pady=10)
row += 1


tk.Label(root, textvariable=result_var, font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, pady=10)


root.mainloop()

