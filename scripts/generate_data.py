import pandas as pd
from src.synthetic_generator import generate_dataset

def main():
    df = generate_dataset()
    
    # save to CSV
    output_path = "data/synthetic/synthetic_delivery_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Generated dataset with {len(df)} rows.")
    print(f"Saved to {output_path}")

    print(df.describe())

if __name__ == "__main__":
    main()