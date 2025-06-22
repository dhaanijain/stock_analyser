import sys
import os
import pandas as pd

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backend'))

from data_analysis import plot_area_chart

# Create a simple test DataFrame
test_data = {
    'date': pd.date_range('2025-01-01', '2025-01-10', freq='D'),
    'Close': [100, 101, 99, 102, 103, 101, 104, 105, 103, 106],
    'sentiment_score': [0.1, 0.2, -0.1, 0.3, 0.4, 0.1, 0.5, 0.6, 0.3, 0.7]
}

df_test = pd.DataFrame(test_data)
print("Test DataFrame created successfully")
print(df_test.head())

try:
    fig = plot_area_chart(df_test)
    print("plot_area_chart function executed successfully!")
except Exception as e:
    print(f"Error in plot_area_chart: {e}") 