import numpy as np

# -----------------------------------------
# Function: compute tip percentage
# this function creates the tip percentage based on correlations observed in real-world data
# -----------------------------------------
def compute_tip_percent(df):
    """
    Compute tip percentage using more realistic, research-informed heuristics.

    Directional assumptions are based on:
    - Normative guidance (15–20% standard for delivery, higher for bad weather,
      long distances, late-night, and large/complex orders).
    - Tipping research showing modest positive effects of service quality and
      negative effects of long waits.
    """

    n = len(df)


    tip = np.full(n, 15.0, dtype=float)   # baseline around standard delivery tip


    # small   orders often hit a "minimum tip" in dollars, so percent goes up.
    small_order = df["order_subtotal"] < 15
    tip[small_order] += 4.0   # push small orders toward ~19%

    # vcry large orders often get slightly higher % as well, but not huge.
    large_order = df["order_subtotal"] > 40
    tip[large_order] += 2.0   # modest bump for big orders

 
    # bad weather means that people are *supposed* to tip more.
    tip += (df["weather"] == "rain") * 2.0
    tip += (df["weather"] == "snow") * 3.0


    # Many people use distance as a fairness cue (more miles → more $).
    # We treat extra miles beyond 4 as adding %.
    extra_dist = np.clip(df["distance_miles"] - 4.0, 0, None) # only count miles beyond 4 
    tip += extra_dist * 0.6   # every extra mile adds ~0.6 percentage points

    # penalize *extra* wait beyond an expected 12 minutes.
    extra_wait = np.clip(df["wait_time_minutes"] - 12.0, 0, None) # only count minutes beyond 12
    tip -= extra_wait * 0.3   # each extra minute beyond 12 cuts tip modestly


    # late night / dinner-type orders: slightly more generous.
    tip += (df["time_of_day"] == "night") * 1.0
    # Early-morning small orders (coffee, etc.) might be a bit less generous.
    tip -= (df["time_of_day"] == "morning") * 0.5


    # weekends (Fri–Sun) cause slightly higher mood & spend.
    weekend = df["day_of_week"].isin(["Fri", "Sat", "Sun"])
    tip += weekend * 0.8

    # Center ratings at 3 (neutral), scale modestly.
    tip += (df["communication_rating"] - 3) * 0.8  # communication matters more

    # Extra items beyond 2 add a small bump.
    extra_items = np.clip(df["item_count"] - 2, 0, None)
    tip += extra_items * 0.25

    # Messages: up to 3 messages add a small positive effect (polite updates).
    msgs_capped = np.minimum(df["messages_sent"], 3)
    tip += msgs_capped * 0.4


    # Real people are inconsistent even with same conditions.
    tip += np.random.normal(0, 3.0, n)

    tip = np.clip(tip, 0, 40)  # 0–40% covers no-tip to very generous but not insane

    return tip


