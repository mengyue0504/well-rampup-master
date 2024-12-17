import argparse
import pandas as pd
import re
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from configuration import StandingInstructionsParsingSettings
from standingInstructionsParsing.standingInstructionsParser import StandingInstructionsParser
import pickle
from configuration import PiAFConnectorSettings

# PI Web API connection settings
PI_WEB_API_URL = "https://piwebapi.osipi.np.woodside/piwebapi/"
USERNAME = "svc.svc.piwebapi"  
PASSWORD = "X9v&k#2ZpR8mU@d4Lq$Y1!oA7%TfWuXQ" 
BASE_PATH = r"GOMPU\Shenzi\Subsea\A. Wells\1.1 Oil Producers"


requests.packages.urllib3.disable_warnings()

def update_attribute_value(webid, value):
    """Update the attribute value in PI AF using the PI Web API."""
    url = f"{PI_WEB_API_URL}/attributes/{webid}/value"
    data = {
        "Timestamp": datetime.now().isoformat(),
        "Value": value
    }
    response = requests.put(url, json=data, auth=HTTPBasicAuth(USERNAME, PASSWORD), verify=False)
    if response.status_code == 204:
        print(f"Successfully updated attribute with WebID {webid}.")
    else:
        print(f"Failed to update attribute with WebID {webid}: {response.status_code}, {response.text}")

# Data parsing and transformation logic
def clean_well_name(well):
    return well.replace(" (cycle)", "").replace("-", "0")

def remove_units(value):
    if isinstance(value, str):
        numeric_part = re.search(r"[-+]?\d*\.?\d+", value)
        if numeric_part:
            print(f"Original value: {value} -> Extracted numeric part: {numeric_part.group()}")
            return float(numeric_part.group())
        else:
            print(f"Warning: Could not extract numeric part from {value}.")
            return None
    return value

def parse_pressure_limits(pressure_limits, well_title):
    limits = {
        "LDHGP": "", "UDHGP": "", "THP": "",
        "DHPT 3/4": "", "DHPT 5/6": "", "DHPT 7/8": "",
        "Press.DH.Lower.Limit": "", "Press.TH.Limit": ""
    }

    if pressure_limits == "N/A" or pressure_limits is None:
        return limits

    limits.update({
        "LDHGP": getattr(pressure_limits, 'LDHGP', ""),
        "UDHGP": getattr(pressure_limits, 'UDHGP', ""),
        "THP": getattr(pressure_limits, 'THP', ""),
        "DHPT 3/4": getattr(pressure_limits, 'DHPT_3_4', ""),
        "DHPT 5/6": getattr(pressure_limits, 'DHPT_5_6', ""),
        "DHPT 7/8": getattr(pressure_limits, 'DHPT_7_8', "")
    })

    limits["Press.TH.Limit"] = limits["THP"]

    for key in ["LDHGP", "UDHGP", "DHPT 3/4", "DHPT 5/6", "DHPT 7/8"]:
        if limits[key]:
            limits["Press.DH.Lower.Limit"] = str(limits[key])
            break

    return limits

def extract_data_to_dataframe(standing_instructions):
    data = {}
    for well, rampup in standing_instructions.rampup_limits.items():
        well_modified = clean_well_name(well)
        pressure_limits_raw = standing_instructions.pressure_limits.get(well, "N/A")
        well_title = well_modified
        pressure_limits = parse_pressure_limits(pressure_limits_raw, well_title)

        data[well_modified] = {
            "Well": well_modified,
            "Ramp-up Limit": str(rampup),
            "LDHGP": pressure_limits["LDHGP"],
            "UDHGP": pressure_limits["UDHGP"],
            "THP": pressure_limits["THP"],
            "DHPT 3/4": pressure_limits["DHPT 3/4"],
            "DHPT 5/6": pressure_limits["DHPT 5/6"],
            "DHPT 7/8": pressure_limits["DHPT 7/8"],
            "Press.DH.Lower.Limit": pressure_limits["Press.DH.Lower.Limit"],
            "Press.TH.Limit": pressure_limits["Press.TH.Limit"]
        }

    df = pd.DataFrame.from_dict(data, orient='index')
    
    df['Press.DH.Lower.Limit'] = df['Press.DH.Lower.Limit'].replace('', 0).astype(float)
    df['Ramp-up Limit'] = df['Ramp-up Limit'].replace('', np.nan) 
    df['Press.TH.Limit'] = df['Press.TH.Limit'].fillna(0) 

    return df

def update_pi_af_attributes(df, df_webids):
    """Update PI AF attributes using WebIDs from the loaded DataFrame."""
    for index, row in df.iterrows():
        well_name = row["Well"]
        rampup_limit = remove_units(row["Ramp-up Limit"])
        th_limit = remove_units(row["Press.TH.Limit"])
        dh_lower_limit = remove_units(row["Press.DH.Lower.Limit"])
        current_entry = datetime.now().strftime("%d-%m-%Y by Python Parser")
        
        webid_row = df_webids[df_webids['Well Name'] == well_name]
        
        if not webid_row.empty:
            # Ramp Up Limit
            rampup_webid = webid_row.iloc[0]['Ramp Up Limit WebID']
            if pd.notna(rampup_webid) and rampup_limit is not None:
                update_attribute_value(rampup_webid, rampup_limit)
                print(f"Successfully updated Ramp Up Limit for {well_name}.")
            
            # Press.TH.Limit
            th_limit_webid = webid_row.iloc[0]['Press.TH.Limit WebID']
            if pd.notna(th_limit_webid) and th_limit is not None:
                update_attribute_value(th_limit_webid, th_limit)
                print(f"Successfully updated Press.TH.Limit for {well_name}.")
            
            # Press.DH.Lower.Limit
            dh_lower_limit_webid = webid_row.iloc[0]['Press.DH.Lower.Limit WebID']
            if pd.notna(dh_lower_limit_webid) and dh_lower_limit is not None:
                update_attribute_value(dh_lower_limit_webid, dh_lower_limit)
                print(f"Successfully updated Press.DH.Lower.Limit for {well_name}.")
            
            # Standing Instructions Date and Entry By
            standing_instructions_webid = webid_row.iloc[0]['Standing Instructions WebID']
            if pd.notna(standing_instructions_webid):
                update_attribute_value(standing_instructions_webid, current_entry)
                print(f"Successfully updated Standing Instructions for {well_name}.")
        else:
            print(f"No WebID entry for well {well_name}, skipping update.")

def main():
    """Main function to parse standing instructions and update PI-AF attributes."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='C:/well-rampup-master/data/standing_instructions')
    parser.add_argument('--well_name_row', type=str, default='production wells')
    parser.add_argument('--pressure_limit_row', type=str, default='flowing pressure limit')
    parser.add_argument('--rampup_limit_row', type=str, default='ramp up limit')
    args = parser.parse_args()

    settings = StandingInstructionsParsingSettings(
        args.path,
        args.well_name_row,
        args.pressure_limit_row,
        args.rampup_limit_row
    )

    try:
        parser = StandingInstructionsParser(settings)
        standing_instructions = parser.read()  # Read data from the input source
        
        df = extract_data_to_dataframe(standing_instructions)
        
        with open('webids_dataframe.pkl', 'rb') as file:
            df_webids = pickle.load(file)

        print("Loaded WebID DataFrame:")
        print(df_webids)

        update_pi_af_attributes(df, df_webids)  # Pass both DataFrames

    except Exception as e:
        error_message = f"Failed to Update: {e}"
        print(error_message)

if __name__ == "__main__":
    main()
