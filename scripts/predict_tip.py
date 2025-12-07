import pandas as pd
import joblib
import argparse

# ======================================================
# Load model + scaler
# ======================================================
model = joblib.load("../models/final_xgboost.pkl")
scaler = joblib.load("../models/final_scaler.pkl")

# ======================================================
# Parse user input
# ======================================================
parser = argparse.ArgumentParser(description="Predict tip percentage for a delivery order.")

parser.add_argument("--distance_miles", type=float, required=True)
parser.add_argument("--order_subtotal", type=float, required=True)
parser.add_argument("--wait_time_minutes", type=float, required=True)

parser.add_argument("--weather", type=str, required=True,
                    choices=["clear", "rain", "snow"])

parser.add_argument("--time_of_day", type=str, required=True,
                    choices=["afternoon", "morning", "night"])

parser.add_argument("--day_of_week", type=str, required=True,
                    choices=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

parser.add_argument("--communication_rating", type=int, required=True)
parser.add_argument("--item_count", type=int, required=True)
parser.add_argument("--messages_sent", type=int, required=True)

args = parser.parse_args()

# ======================================================
# Build correctly encoded input row
# ======================================================
data = {
    "distance_miles": [args.distance_miles],
    "order_subtotal": [args.order_subtotal],
    "wait_time_minutes": [args.wait_time_minutes],
    "communication_rating": [args.communication_rating],
    "item_count": [args.item_count],
    "messages_sent": [args.messages_sent],

    # Weather (base = clear)
    "weather_rain": [1 if args.weather == "rain" else 0],
    "weather_snow": [1 if args.weather == "snow" else 0],

    # Time of day (base = afternoon)
    "time_of_day_morning": [1 if args.time_of_day == "morning" else 0],
    "time_of_day_night": [1 if args.time_of_day == "night" else 0],

    # Day of week (base = Fri)
    "day_of_week_Mon": [1 if args.day_of_week == "Mon" else 0],
    "day_of_week_Sat": [1 if args.day_of_week == "Sat" else 0],
    "day_of_week_Sun": [1 if args.day_of_week == "Sun" else 0],
    "day_of_week_Thu": [1 if args.day_of_week == "Thu" else 0],
    "day_of_week_Tue": [1 if args.day_of_week == "Tue" else 0],
    "day_of_week_Wed": [1 if args.day_of_week == "Wed" else 0],
}

df = pd.DataFrame(data)

# ======================================================
# Scale features
# ======================================================
X_scaled = scaler.transform(df)

# ======================================================
# Predict tip %
# ======================================================
prediction = model.predict(X_scaled)[0]

print("\n=================================")

print(f"Predicted Tip Percentage: {prediction:.2f}%")
# print out the predicted tip amount in dollars
tip_amount = (prediction / 100) * args.order_subtotal
print(f"Predicted Tip Amount: ${tip_amount:.2f}")
print("=================================\n")

# run it like this with the parameters filled in:
'''python3 predict_tip.py \
  --distance_miles 2.5 \
  --order_subtotal 25.00 \
  --wait_time_minutes 12 \
  --weather clear \
  --time_of_day night \
  --day_of_week Mon \
  --communication_rating 4 \
  --item_count 3 \
  --messages_sent 1

'''