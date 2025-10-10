import requests
import os
import pandas as pd
import re
import time
import zipfile
import gzip
import json
from tqdm import tqdm

# Utility function to flatten and join list elements
def flatten_and_join(value):
    """Flattens list elements and joins them into a single string."""
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(str(item))
        return ' '.join(flattened)
    elif isinstance(value, str):
        return value
    else:
        return str(value)

# Step 1: Download JSON files from PubChem FTP
def download_pubchem_json_files(save_directory="./DB/PubChem/pubchem_bioassay_json"):
    """
    Downloads all Bioassay JSON files from the PubChem FTP server and saves them as ZIP files.
    
    Args:
        save_directory (str): Directory to save downloaded ZIP files.
    """
    ftp_base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/JSON/"
    os.makedirs(save_directory, exist_ok=True)

    response = requests.get(ftp_base_url)
    if response.status_code != 200:
        print("Failed to retrieve the directory listing from FTP server.")
        return

    zip_files = re.findall(r'href=["\'](\d+_\d+\.zip)["\']', response.text)
    for zip_file in tqdm(zip_files, desc="Downloading ZIP files"):
        zip_url = f"{ftp_base_url}{zip_file}"
        local_zip_path = os.path.join(save_directory, zip_file)
        if not os.path.exists(local_zip_path):
            with requests.get(zip_url, stream=True) as r:
                if r.status_code == 200:
                    with open(local_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Failed to download {zip_url}")
    print("All ZIP files downloaded.")

# Step 2: Extract AID metadata and MoA
def process_aid_metadata(aid, aid_to_uniprot_gene_name, json_dir="./DB/PubChem/pubchem_bioassay_json"):
    """
    Extracts metadata (Gene_Name, MoA, etc.) for a given AID from local JSON files.
    
    Args:
        aid (int): PubChem Assay ID.
        aid_to_uniprot_gene_name (dict): Mapping of AID to UNIPROT_GENE_NAME.
        json_dir (str): Directory containing JSON ZIP files.
    
    Returns:
        dict: Extracted metadata or None if failed.
    """
    start = ((aid - 1) // 1000) * 1000 + 1
    end = start + 999
    zip_file_name = f"{start:07d}_{end:07d}.zip"
    zip_file_path = os.path.join(json_dir, zip_file_name)
    json_file_name = f"{aid}.json.gz"

    if not os.path.exists(zip_file_path):
        print(f"Zip file {zip_file_path} does not exist.")
        return None

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zip_file_list = zf.namelist()
            found_name = next((name for name in zip_file_list if name.endswith(f"/{json_file_name}") or name.endswith(f"\\{json_file_name}") or name == json_file_name), None)
            if found_name is None:
                print(f"JSON file {json_file_name} not found in zip file {zip_file_name}")
                return None
            with zf.open(found_name) as json_file:
                with gzip.open(json_file, 'rt', encoding='utf-8') as gz:
                    data = json.load(gz)
    except (zipfile.BadZipFile, KeyError, Exception) as e:
        print(f"Error reading AID {aid}: {e}")
        return None

    try:
        assay_descr = data['PC_AssaySubmit']['assay']['descr']
        assay_name = flatten_and_join(assay_descr.get('name', ''))
        assay_description = flatten_and_join(assay_descr.get('description', ''))
        assay_comments = flatten_and_join(assay_descr.get('comment', ''))

        gene_name = ""
        taxonomy = "Not specified"
        accession = "Not specified"
        if "target" in assay_descr and assay_descr['target']:
            target = assay_descr['target'][0]
            gene_name = target.get('name', '')
            mol_id = target.get('mol_id', {})
            accession = mol_id.get('protein_accession', 'Not specified')
            organism = target.get('organism', {})
            if organism:
                org = organism.get('org', {})
                taxonomy = org.get('taxname', 'Not specified')

        target_receptor = aid_to_uniprot_gene_name.get(aid, '').lower() or ''

        moa = 'unknown'
        text_to_search = ' '.join([assay_name, assay_description, assay_comments])
        sentences = re.split(r'(?<=[.!?])\s+', text_to_search)
        moa_terms = [
            ('positive allosteric modulator', 'positive allosteric modulator'),
            ('negative allosteric modulator', 'negative allosteric modulator'),
            ('inverse agonist', 'inverse agonist'),
            ('partial agonist', 'partial agonist'),
            ('antagonist', 'antagonist'),
            ('agonist', 'agonist')
        ]

        for term, moa_label in moa_terms:
            for sentence in sentences:
                if target_receptor in sentence.lower() and re.search(rf'\b{term}\b', sentence, re.IGNORECASE):
                    moa = moa_label
                    break
            if moa != 'unknown':
                break

        active_criteria = 'Not specified'
        match = re.search(r'(\d+\.?\d*)\s*(nM|μM|uM|mM)', text_to_search, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            active_criteria = f"{value}{unit}"

        return {
            'AID': aid, 'Gene_Name': gene_name, 'Target_Receptor': target_receptor,
            'Accession': accession, 'Taxonomy': taxonomy, 'MoA': moa, 'Active_Criteria': active_criteria
        }
    except (KeyError, IndexError):
        print(f"JSON structure unexpected for AID {aid}")
        return None

# Step 3: Extract activity data (IC50, EC50, etc.)
def process_aid_activity(aid, failed_aids, json_dir="./DB/PubChem/pubchem_bioassay_json"):
    """
    Extracts activity data (IC50, EC50, etc.) for a given AID from local JSON files.
    
    Args:
        aid (int): PubChem Assay ID.
        failed_aids (list): List to store failed AIDs.
        json_dir (str): Directory containing JSON ZIP files.
    
    Returns:
        list: List of dictionaries containing activity data.
    """
    start = ((aid - 1) // 1000) * 1000 + 1
    end = start + 999
    zip_file_name = f"{start:07d}_{end:07d}.zip"
    zip_file_path = os.path.join(json_dir, zip_file_name)
    json_file_name = f"{aid}.json.gz"

    if not os.path.exists(zip_file_path):
        print(f"Zip file {zip_file_path} does not exist.")
        failed_aids.append(aid)
        return []

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zip_file_list = zf.namelist()
            found_name = next((name for name in zip_file_list if name.endswith(f"/{json_file_name}") or name.endswith(f"\\{json_file_name}") or name == json_file_name), None)
            if found_name is None:
                print(f"JSON file {json_file_name} not found in zip file {zip_file_name}")
                failed_aids.append(aid)
                return []
            with zf.open(found_name) as json_file:
                with gzip.open(json_file, 'rt', encoding='utf-8') as gz:
                    data = json.load(gz)
    except (zipfile.BadZipFile, KeyError, Exception) as e:
        print(f"Error reading AID {aid}: {e}")
        failed_aids.append(aid)
        return []

    try:
        assay_descr = data['PC_AssaySubmit']['assay']['descr']
        assay_data = data['PC_AssaySubmit']['data']
        unit_mapping = {5: 'uM', 6: 'nM', 7: 'mM', 8: 'pM', 9: 'fM', 10: 'M', 254: 'None'}
        aid_data = []

        for entry in assay_data:
            sid = entry.get('sid', 'Not specified')
            activity_outcome = entry.get('outcome', 'Not specified')
            activity = 2 if activity_outcome == 2 else 1

            selected_metric = None
            metrics_priority_lower = [m.lower() for m in [
                'IC50', 'EC50', 'Ki', 'Kd', 'pIC50', 'pEC50', 'pKi', 'pKd', 'LogIC50', 'LogEC50', 'LogKi', 'LogKd'
            ]]

            for result in entry.get('data', []):
                tid_index = result['tid'] - 1
                if tid_index < len(assay_descr['results']):
                    tid_name = assay_descr['results'][tid_index]['name'].lower()
                    if tid_name in metrics_priority_lower:
                        value = result['value'].get('fval') if 'fval' in result['value'] else result['value'].get('sval', 'Not specified')
                        unit_code = assay_descr['results'][tid_index].get('unit', 'Not specified')
                        unit = unit_mapping.get(unit_code, 'Unknown')

                        if (selected_metric is None or metrics_priority_lower.index(tid_name) < metrics_priority_lower.index(selected_metric['Type'].lower())):
                            selected_metric = {
                                'AID': aid, 'SID': sid, 'Activity': activity, 'Type': tid_name,
                                'Value': value, 'Unit': unit
                            }

            if selected_metric:
                aid_data.append(selected_metric)

        return aid_data
    except KeyError as e:
        print(f"JSON structure unexpected for AID {aid}: Missing key {e}")
        failed_aids.append(aid)
        return []

# Main execution
def main():
    """Execute the full PubChem Bioassay parsing pipeline."""
    print("Starting PubChem Bioassay data mining...")

    # Step 1: Download JSON files
    download_pubchem_json_files()

    # Step 2: Load AID mapping and process metadata
    df = pd.read_csv('./DB/PubChem/tb_aid_act_gpcr.csv', dtype={'GENE_ID': str})
    aid_to_uniprot_gene_name = df.dropna(subset=['UNIPROT_GENE_NAME']).set_index('AID')['UNIPROT_GENE_NAME'].str.lower().to_dict()
    aids = df['AID'].unique().tolist()

    metadata_data = []
    for aid in tqdm(aids, desc="Extracting AID metadata"):
        result = process_aid_metadata(aid, aid_to_uniprot_gene_name)
        if result:
            metadata_data.append(result)
        time.sleep(0.2)

    metadata_df = pd.DataFrame(metadata_data)
    metadata_df.to_csv('./Output/DB/PubChem/aid_information.csv', index=False)
    print(f"Metadata extracted for {len(metadata_df)} AIDs.")

    # Step 3: Extract activity data
    activity_data = []
    failed_aids = []
    for aid in tqdm(aids, desc="Extracting activity data"):
        result = process_aid_activity(aid, failed_aids)
        activity_data.extend(result)
        time.sleep(0.2)

    activity_df = pd.DataFrame(activity_data)
    activity_df.to_csv('./Output/DB/PubChem/aid_activity_data.csv', index=False)
    print(f"Activity data extracted for {len(activity_df)} entries. Failed AIDs: {len(failed_aids)}")

if __name__ == "__main__":
    main()