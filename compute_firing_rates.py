"""
Neural Firing Rate Processor
----------------------------
Batch processes neural recording data to compute firing rates for specific brain regions.

Logic:
    1. Loads brain region mapping.
    2. Identifies target CSV files based on date patterns.
    3. Filters for 'Premotor' cortex channels.
    4. Downsamples spike data (summing spikes over 300ms bins) and converts to Hz.

Author: Yu Puzhi
Last Updated: 2025-10-1
"""

import os
import glob
import re
import numpy as np
import pandas as pd

# --- Configuration & Constants ---
# Paths (It is best practice to allow these to be easily changed)
REGION_INFO_PATH = r'D:\class_material\monkey_data\brain_region\omg-P-eve-boxcar-norm-231003.csv'
DATA_DIR = r'D:\class_material\pacman-transformer2biology\results\results'
OUTPUT_SUFFIX = "_firing_rate_PMC.csv"

# Signal Processing Constants
TARGET_REGION = 'Premotor'
BIN_SIZE_MS = 300           # Target bin size
ROW_DURATION_MS = 100 / 6   # Raw sampling rate: ~16.667 ms per row

# Calculate samples per bin: 300ms / (100/6)ms = 18 samples
SAMPLES_PER_BIN = int(BIN_SIZE_MS / ROW_DURATION_MS) 
BIN_DURATION_SEC = BIN_SIZE_MS / 1000.0


def calculate_firing_rate(spike_series, samples_per_group=18, duration_sec=0.3):
    """
    Computes firing rate (Hz) by binning spikes and normalizing by time.
    
    Args:
        spike_series (np.array): Raw spike data (binary or counts).
        samples_per_group (int): Number of rows to aggregate (window size).
        duration_sec (float): Duration of the window in seconds for normalization.
        
    Returns:
        np.array: Firing rate in Hz, upsampled back to original length.
    """
    spike_data = np.nan_to_num(spike_series) # Handle NaNs safely
    n_rows = len(spike_data)
    
    # distinct valid rows that fit into the bin size
    n_valid_rows = n_rows - (n_rows % samples_per_group)
    
    if n_valid_rows == 0:
        return np.full(n_rows, np.nan)

    # Reshape to (N_bins, 18) and sum spikes within each bin
    valid_data = spike_data[:n_valid_rows]
    binned_sum = valid_data.reshape(-1, samples_per_group).sum(axis=1)
    
    # Calculate Hz: (Sum of spikes) / (0.3s)
    hz_rate = binned_sum / duration_sec
    
    # Expand back to original resolution (step function)
    expanded_rate = np.repeat(hz_rate, samples_per_group)
    
    # Pad remainder with NaN if total length is not divisible by 18
    remainder = n_rows % samples_per_group
    if remainder > 0:
        expanded_rate = np.concatenate([expanded_rate, np.full(remainder, np.nan)])
        
    return expanded_rate


def main():
    # 1. Load Brain Region Metadata
    try:
        df_info = pd.read_csv(REGION_INFO_PATH)
        print(f"Metadata loaded. Total records: {len(df_info)}")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {REGION_INFO_PATH}")
        return

    # 2. Locate Data Files
    search_pattern = os.path.join(DATA_DIR, "*pFlip.csv")
    file_list = glob.glob(search_pattern)
    print(f"Found {len(file_list)} files to process.")

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        # A. Extract Date from Filename (Format: 01-Apr-2021)
        date_match = re.search(r'\d{2}-[A-Za-z]{3}-\d{4}', file_name)
        if not date_match:
            print(f"[SKIP] {file_name}: Date pattern not found.")
            continue
        
        current_date = date_match.group()

        # B. Identify Relevant Channels (Premotor)
        # Filter metadata for current date and target region
        mask = (df_info['date'] == current_date) & (df_info['brain_areas'] == TARGET_REGION)
        target_channels = df_info[mask]['ch'].astype(str).tolist()
        
        if not target_channels:
            print(f"[SKIP] {file_name}: No '{TARGET_REGION}' channels found for {current_date}.")
            continue

        # C. Load Raw Data
        df_raw = pd.read_csv(file_path)
        df_raw.columns = df_raw.columns.astype(str) # Ensure string matching for columns
        
        # Intersect available columns with target channels
        available_cols = [col for col in target_channels if col in df_raw.columns]
        
        if not available_cols:
            print(f"[SKIP] {file_name}: No matching channel columns in file.")
            continue

        # D. Process Data
        # Initialize output DataFrame. Preserve 'Index' if exists, else create range.
        output_df = pd.DataFrame()
        output_df['Index'] = df_raw.get('Index', np.arange(len(df_raw)))

        for col in available_cols:
            output_df[col] = calculate_firing_rate(
                df_raw[col].values, 
                samples_per_group=SAMPLES_PER_BIN, 
                duration_sec=BIN_DURATION_SEC
            )

        # E. Save Results
        output_name = file_name.replace(".csv", OUTPUT_SUFFIX)
        output_path = os.path.join(DATA_DIR, output_name)
        output_df.to_csv(output_path, index=False)
        
        print(f"[OK] Processed {file_name} -> {output_name} ({len(available_cols)} channels)")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()