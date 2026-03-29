"""
Script to generate additional synthetic data for fish farm disease prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

def generate_synthetic_data(n_samples=2000, output_file=None):
    """
    Generate synthetic fish farm data with realistic patterns
    """
    np.random.seed(42)
    
    # Read existing data to understand patterns
    existing_data = pd.read_csv(settings.RAW_DATA_FILE)
    last_date = pd.to_datetime(existing_data['Date'].iloc[-1])
    
    print(f"Generating {n_samples} new data rows...")
    print(f"Starting from date: {last_date + timedelta(days=1)}")
    
    # Generate dates
    dates = [last_date + timedelta(days=i+1) for i in range(n_samples)]
    
    # Initialize lists
    data = {
        'Date': [d.strftime('%Y-%m-%d') for d in dates],  # Format dates consistently
        'Temperature_C': [],
        'pH': [],
        'Dissolved_Oxygen_mg_L': [],
        'Ammonia_mg_L': [],
        'Nitrate_mg_L': [],
        'Turbidity_NTU': [],
        'Feed_Intake_Percent': [],
        'Growth_Rate_g_week': [],
        'Mortality_Count_per_day': [],
        'Disease_Outbreak': []
    }
    
    for i in range(n_samples):
        # Decide if this will be an outbreak day (30% probability)
        is_outbreak = np.random.random() < 0.30
        
        if is_outbreak:
            # Critical conditions leading to outbreak
            temp = np.random.choice([
                np.random.uniform(18, 22),  # Too cold
                np.random.uniform(30, 34)   # Too hot
            ])
            ph = np.random.choice([
                np.random.uniform(5.5, 6.3),  # Too acidic
                np.random.uniform(8.2, 8.5)    # Too alkaline
            ])
            do = np.random.uniform(2.0, 4.5)  # Low oxygen
            ammonia = np.random.uniform(1.0, 4.5)  # High ammonia
            nitrate = np.random.uniform(50, 95)  # High nitrate
            turbidity = np.random.uniform(60, 95)  # High turbidity
            feed_intake = np.random.uniform(30, 60)  # Low feed intake
            growth_rate = np.random.uniform(0.3, 0.9)  # Poor growth
            mortality = np.random.randint(10, 40)  # High mortality
            
        else:
            # Healthy conditions
            temp = np.random.normal(26, 2)
            temp = np.clip(temp, 23, 29)
            
            ph = np.random.normal(7.0, 0.3)
            ph = np.clip(ph, 6.5, 7.8)
            
            do = np.random.normal(7.5, 1.5)
            do = np.clip(do, 5.5, 9.5)
            
            ammonia = np.random.exponential(0.3)
            ammonia = np.clip(ammonia, 0.01, 0.8)
            
            nitrate = np.random.uniform(5, 35)
            
            turbidity = np.random.uniform(10, 50)
            
            feed_intake = np.random.normal(85, 10)
            feed_intake = np.clip(feed_intake, 65, 100)
            
            growth_rate = np.random.normal(1.7, 0.3)
            growth_rate = np.clip(growth_rate, 1.0, 2.5)
            
            mortality = np.random.poisson(4)
            mortality = np.clip(mortality, 0, 8)
        
        # Add seasonal variations
        day_of_year = dates[i].timetuple().tm_yday
        seasonal_temp_adjust = 2 * np.sin(2 * np.pi * day_of_year / 365)
        temp += seasonal_temp_adjust
        temp = np.clip(temp, 18, 33)
        
        # Store values
        data['Temperature_C'].append(round(temp, 2))
        data['pH'].append(round(ph, 2))
        data['Dissolved_Oxygen_mg_L'].append(round(do, 2))
        data['Ammonia_mg_L'].append(round(ammonia, 2))
        data['Nitrate_mg_L'].append(round(nitrate, 2))
        data['Turbidity_NTU'].append(round(turbidity, 2))
        data['Feed_Intake_Percent'].append(round(feed_intake, 1))
        data['Growth_Rate_g_week'].append(round(growth_rate, 2))
        data['Mortality_Count_per_day'].append(int(mortality))
        data['Disease_Outbreak'].append(1 if is_outbreak else 0)
    
    # Create DataFrame
    new_data = pd.DataFrame(data)
    
    # Combine with existing data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Save to file
    if output_file is None:
        output_file = settings.RAW_DATA_FILE
    
    combined_data.to_csv(output_file, index=False)
    
    print(f"\n✓ Data generation complete!")
    print(f"  Total rows: {len(combined_data)}")
    print(f"  Outbreak rate: {combined_data['Disease_Outbreak'].mean()*100:.1f}%")
    print(f"  Saved to: {output_file}")
    
    return combined_data

if __name__ == '__main__':
    generate_synthetic_data(n_samples=2000)
