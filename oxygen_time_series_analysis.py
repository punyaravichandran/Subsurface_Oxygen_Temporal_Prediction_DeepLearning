"""
Time Series Oxygen Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Step 1: Load Dataset
file_path = "combine_data.csv"  
data = pd.read_csv(file_path)

# Convert to datetime
data['SAMPLE_DATE'] = pd.to_datetime(
    data['Date'],
    format='%Y-%m-%dT%H:%M:%S.%fZ',
    errors='coerce'
)

# Drop original Date column
data = data.drop(columns=['Date'])

# Step 2: Select & Clean Data
final_sc = data[['Oxygen', 'SAMPLE_DATE']].copy()

# Interpolate missing Oxygen values
final_sc['Oxygen'] = final_sc['Oxygen'].interpolate(method='linear')

# Drop remaining NaNs
final_sc = final_sc.dropna()


# Step 3: Filter Date Range
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2026-01-01')

final_sc = final_sc[
    (final_sc['SAMPLE_DATE'] >= start_date) &
    (final_sc['SAMPLE_DATE'] <= end_date)
]

# Step 4: Monthly Time-Series Stats
final_sc['Month_Year'] = final_sc['SAMPLE_DATE'].dt.to_period('M')

monthly_stats = (
    final_sc.groupby('Month_Year')['Oxygen']
    .agg(['mean', 'std'])
    .reset_index()
)

monthly_stats.rename(columns={
    'mean': 'Mean_Oxygen',
    'std': 'Std_Oxygen'
}, inplace=True)

monthly_stats['Month_Year'] = monthly_stats['Month_Year'].dt.to_timestamp()

# Step 5: Plot Time-Series
plt.figure(figsize=(15, 7))

sns.lineplot(
    x='Month_Year',
    y='Mean_Oxygen',
    data=monthly_stats,
    marker='o',
    label='Mean Oxygen'
)

plt.fill_between(
    monthly_stats['Month_Year'],
    monthly_stats['Mean_Oxygen'] - monthly_stats['Std_Oxygen'],
    monthly_stats['Mean_Oxygen'] + monthly_stats['Std_Oxygen'],
    alpha=0.2,
    label='Std Dev'
)

plt.title('Monthly Mean Oxygen Concentration')
plt.xlabel('Time')
plt.ylabel('Oxygen Concentration')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

