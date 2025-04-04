import pandas as pd
import os
from datetime import datetime, timedelta

def load_stock_data(data_dir):
    stock_files = {
        'Apple': 'apple_processed.csv',
        'Amazon': 'amazon_processed.csv',
        'Google': 'google_processed.csv',
        'Microsoft': 'microsoft_processed.csv',
        'Tesla': 'tesla_processed.csv'
    }

    stock_data = {}
    five_years_ago = datetime.now() - timedelta(days=5*365)

    for name, filename in stock_files.items():
        file_path = os.path.join(data_dir, filename)
        print(f"\nLoading {name} from {file_path}")

        # Load raw CSV
        df = pd.read_csv(file_path)

        # Clean column names
        df.columns = df.columns.str.strip().str.title()

        # Combine Date + Time and parse to datetime
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df.set_index('Datetime', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)

        # Filter to only include data from the last 5 years
        df = df[df.index >= five_years_ago]

        if df.empty:
            print(f"{name} has no data in the last 5 years. Skipping.")
            continue
        
        print(f"{name}: {len(df)} rows after filtering for the last 5 years")

        # Store cleaned DataFrame
        stock_data[name] = df

    return stock_data
