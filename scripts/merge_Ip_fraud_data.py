import pandas as pd
import numpy as np
import struct
import socket

class MergeIPFraudData:
    """
    A class to merge fraud data with IP address geolocation data.
    """

    def __init__(self, fraud_data_path, ipaddress_data_path, output_path):
        self.fraud_data_path = fraud_data_path
        self.ipaddress_data_path = ipaddress_data_path
        self.output_path = output_path

    @staticmethod
    def ip_to_int(ip):
        """Convert an IP address to its integer representation."""
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except socket.error:
            return None  # Handle invalid IPs gracefully

    def merge_fraud_ip_data(self):
        """
        Merge fraud data with IP address geolocation data based on IP ranges.
        """
        # Load data
        fraud_data = pd.read_csv(self.fraud_data_path)
        ipaddress_data = pd.read_csv(self.ipaddress_data_path)

        # Convert IP addresses in fraud data to integer format
        fraud_data['ip_int'] = fraud_data['ip_address'].apply(
            lambda x: self.ip_to_int(str(x)) if pd.notna(x) else None
        )

        # Convert bounds in the country data to int64
        ipaddress_data['lower_bound_ip_address'] = ipaddress_data['lower_bound_ip_address'].astype(np.int64)
        ipaddress_data['upper_bound_ip_address'] = ipaddress_data['upper_bound_ip_address'].astype(np.int64)

        # Ensure fraud_data's ip_int is int64 as well
        fraud_data.dropna(subset=['ip_int'], inplace=True)  # Drop rows with invalid IPs
        fraud_data['ip_int'] = fraud_data['ip_int'].astype(np.int64)

        # Sort both datasets for merge_asof
        fraud_data.sort_values('ip_int', inplace=True)
        ipaddress_data.sort_values('lower_bound_ip_address', inplace=True)

        # Merge the datasets using merge_asof
        merged_data = pd.merge_asof(
            fraud_data,
            ipaddress_data,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        # Filter rows where ip_int is within the lower and upper bounds
        merged_data = merged_data[
            (merged_data['ip_int'] >= merged_data['lower_bound_ip_address']) &
            (merged_data['ip_int'] <= merged_data['upper_bound_ip_address'])
        ]

        # Drop unnecessary columns
        merged_data.drop(columns=['ip_address', 'lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

        # Save to CSV file
        merged_data.to_csv(self.output_path, index=False)
        
        print(f"Merged data saved to {self.output_path}")

# Running in a Jupyter Notebook:
fraud_file = "fraud_data.csv"
ip_file = "ip_geolocation_data.csv"
output_file = "merged_fraud_ip_data.csv"

# Create an instance and run the function
merger = MergeIPFraudData(fraud_file, ip_file, output_file)
merger.merge_fraud_ip_data()
