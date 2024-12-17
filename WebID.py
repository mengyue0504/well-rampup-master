import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import pickle
from configuration import PiConnectorSettings

# PI Web API connection settings
PI_WEB_API_URL = "https://piwebapi.osipi.np.woodside/piwebapi/"
USERNAME = "svc.svc.piwebapi"  
PASSWORD = "X9v&k#2ZpR8mU@d4Lq$Y1!oA7%TfWuXQ"  
BASE_PATH = r"GOMPU\Shenzi\Subsea\A. Wells\1.1 Oil Producers"

requests.packages.urllib3.disable_warnings()

def get_webid_by_path(path, object_type="element"):
    """
    Retrieve the WebID by the path for an element or attribute.
    """
    AUTH = (USERNAME, PASSWORD)
    url = f"{PI_WEB_API_URL}/{object_type}s?path={path}"
    response = requests.get(url, auth=AUTH, verify=False)
    if response.status_code == 200:
        webid = response.json().get('WebId')
        return webid
    else:
        print(f"Failed to retrieve WebID for {path}: {response.status_code}, {response.text}")
        return None

def get_elements_under_path(web_api_url, parent_webid):
    """
    Retrieve all elements under the specified path by the parent WebID.
    """
    url = f"{web_api_url}elements/{parent_webid}/elements"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), verify=False)
    if response.status_code == 200:
        elements = response.json()['Items']
        return elements
    else:
        print(f"Error: {response.status_code} - Could not retrieve elements. Response: {response.text}")
        return None

def main():
    """
    Main function to list all elements under the specified path and retrieve WebIDs for various attributes.
    """
    target_path = r"\\AWSENT-AF\Shenzi Ramp-Up\GOMPU\Shenzi\Subsea\A. Wells\1.1 Oil Producers"
    
    # First, get the WebID for the target path
    parent_webid = get_webid_by_path(target_path)
    
    if parent_webid:
        # Fetch all elements under this path
        elements = get_elements_under_path(PI_WEB_API_URL, parent_webid)
        
        if elements:
            # Create a list to hold the data
            data = []
            
            for element in elements:
                well_name = element['Name']
                
                # Paths for the various attributes
                ramp_up_limit_path = f"{element['Path']}|Ramp Up Limit (psi/hr)"
                press_th_limit_path = f"{element['Path']}|Press.TH.Limit"
                press_dh_lower_limit_path = f"{element['Path']}|Press.DH.Lower.Limit"
                standing_instructions_path = f"{element['Path']}|Standing Instructions Date and Entry By"
                
                # Retrieve WebIDs for the attributes
                ramp_up_limit_webid = get_webid_by_path(ramp_up_limit_path, object_type="attribute")
                press_th_limit_webid = get_webid_by_path(press_th_limit_path, object_type="attribute")
                press_dh_lower_limit_webid = get_webid_by_path(press_dh_lower_limit_path, object_type="attribute")
                standing_instructions_webid = get_webid_by_path(standing_instructions_path, object_type="attribute")
                
                if ramp_up_limit_webid and press_th_limit_webid and press_dh_lower_limit_webid and standing_instructions_webid:
                    print(f"WebIDs for '{well_name}': Ramp Up Limit: {ramp_up_limit_webid}, Press.TH.Limit: {press_th_limit_webid}, Press.DH.Lower.Limit: {press_dh_lower_limit_webid}, Standing Instructions: {standing_instructions_webid}")
                    # Append the well name and WebIDs to the data list
                    data.append({
                        "Well Name": well_name,
                        "Ramp Up Limit WebID": ramp_up_limit_webid,
                        "Press.TH.Limit WebID": press_th_limit_webid,
                        "Press.DH.Lower.Limit WebID": press_dh_lower_limit_webid,
                        "Standing Instructions WebID": standing_instructions_webid
                    })
                else:
                    print(f"WebID(s) not found for {well_name}, skipping.")
            
            # Convert the data into a Pandas DataFrame
            df_webids = pd.DataFrame(data)
            
            # Display the DataFrame
            print(df_webids)
            
            # Save the DataFrame to a file using pickle
            with open('webids_dataframe.pkl', 'wb') as file:
                pickle.dump(df_webids, file)

            print("WebID DataFrame saved to 'webids_dataframe.pkl'.")
        else:
            print("No elements found under the target path.")
    else:
        print("Failed to retrieve parent WebID for the target path.")

if __name__ == "__main__":
    main()