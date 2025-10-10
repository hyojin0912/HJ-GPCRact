###########################
#### For DrugBank, ChEMBL, and BindingDB mining
###########################
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
import sqlite3
import chembl_downloader

# Function to parse DrugBank XML data
def parse_drugbank(xml_file_path='./DB/DrugBank/full_database.xml'):
    """
    Parse DrugBank XML file to extract drug-target interaction data with MoA annotations.
    
    Args:
        xml_file_path (str): Path to the DrugBank XML file.
    
    Returns:
        pd.DataFrame: DataFrame containing DrugBankID, Target UNIPROT_AC, and Action Type.
    """
    # Define DrugBank namespace
    ns = {"drugbank": "http://www.drugbank.ca"}
    
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract required information
    data = []
    for drug in root.findall("drugbank:drug", ns):
        drugbank_id = drug.find("drugbank:drugbank-id[@primary='true']", ns).text
        
        # Loop through each target for this drug
        for target in drug.findall("drugbank:targets/drugbank:target", ns):
            uniprot_ac_elem = target.find("drugbank:polypeptide", ns)
            uniprot_ac = uniprot_ac_elem.get("id") if uniprot_ac_elem is not None else "N/A"
            
            # Extract all action types for each target
            action_types = [action.text for action in target.findall("drugbank:actions/drugbank:action", ns)]
            
            # Capture entries for each action type
            for action in action_types:
                data.append({
                    "DrugBankID": drugbank_id,
                    "Target UNIPROT_AC": uniprot_ac,
                    "Action Type": action
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('./Output/DB/DrugBank/DrugBank_ActionType_xml.csv', index=None)
    print(f"Total rows extracted from DrugBank: {len(df)}")
    print(f"Unique DrugBankIDs with action types: {len(df['DrugBankID'].unique())}")
    return df

# Function to parse ChEMBL SQLite database
def parse_chembl():
    """
    Parse ChEMBL SQLite database to extract drug-target interaction data with MoA and assay information.
    
    Returns:
        pd.DataFrame: DataFrame containing mechanism and assay data.
    """
    # Download and extract ChEMBL SQLite database
    path = chembl_downloader.download_extract_sqlite(version='34')
    conn = sqlite3.connect(path)
    
    # Query 1: Extract mechanism data
    with conn:
        cursor = conn.cursor()
        sql_mechanism = """
        SELECT
            DRUG_MECHANISM.molregno, DRUG_MECHANISM.tid, TARGET_DICTIONARY.chembl_id,
            DRUG_MECHANISM.mechanism_of_action, DRUG_MECHANISM.action_type,
            COMPOUND_STRUCTURES.standard_inchi_key, COMPOUND_STRUCTURES.canonical_smiles
        FROM DRUG_MECHANISM
        JOIN TARGET_DICTIONARY ON DRUG_MECHANISM.tid = TARGET_DICTIONARY.tid
        JOIN COMPOUND_STRUCTURES ON DRUG_MECHANISM.molregno = COMPOUND_STRUCTURES.molregno
        """
        cursor.execute(sql_mechanism)
        rows_mechanism = cursor.fetchall()
        df_mechanism = pd.DataFrame(rows_mechanism, columns=[desc[0] for desc in cursor.description])
        print("Sample mechanism data from ChEMBL:")
        print(df_mechanism.head())
    
    # Query 2: Extract assay data
    sql_assay = """
    SELECT 
        ACTIVITIES.MOLREGNO, ASSAYS.TID, TARGET_DICTIONARY.CHEMBL_ID,
        ACTIVITIES.STANDARD_TYPE, ACTIVITIES.STANDARD_RELATION, ACTIVITIES.STANDARD_VALUE, ACTIVITIES.STANDARD_UNITS,
        ASSAYS.DESCRIPTION,
        COMPOUND_STRUCTURES.STANDARD_INCHI_KEY, COMPOUND_STRUCTURES.CANONICAL_SMILES
    FROM ACTIVITIES
    JOIN ASSAYS ON ACTIVITIES.ASSAY_ID = ASSAYS.ASSAY_ID
    JOIN COMPOUND_STRUCTURES ON ACTIVITIES.MOLREGNO = COMPOUND_STRUCTURES.MOLREGNO
    JOIN TARGET_DICTIONARY ON ASSAYS.TID = TARGET_DICTIONARY.TID
    WHERE RELATIONSHIP_TYPE = 'D' AND TARGET_TYPE = 'SINGLE PROTEIN' 
    AND (STANDARD_UNITS = 'nM' OR STANDARD_UNITS = 'uM')
    """
    df_assay = chembl_downloader.query(sql_assay)
    df_assay.to_csv('/home/users/hyojin0912/Activity_v2/DB/ChEMBL/ASSAYS_SINGLE_DIRECT_v34.csv', index=None)
    
    # List available tables for verification
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Available tables in ChEMBL database:")
        for table in tables:
            print(table[0])
    
    return pd.concat([df_mechanism, df_assay], axis=0, ignore_index=True)

# Function to parse BindingDB data
def parse_bindingdb():
    """
    Parse BindingDB TSV files to extract GPCR-specific binding affinity data.
    
    Returns:
        pd.DataFrame: DataFrame containing GPCR binding assay data.
    """
    # Define columns to load from BindingDB files
    columns_to_use = ['BindingDB Reactant_set_id', 'PubChem CID', 'Kd (nM)', 'Ki (nM)', 'IC50 (nM)', 'EC50 (nM)', 
                      'Ligand SMILES', 'Ligand InChI Key', 'UniProt (SwissProt) Primary ID of Target Chain']
    bindingdb_all = pd.read_csv('./DB/BindingDB/BindingDB_All.tsv', sep='\t', usecols=columns_to_use, low_memory=False) # from BindingDB web download service
    bdb_rsid_eaids = pd.read_csv('./DB/BindingDB/BDB_rsid_eaids.tsv', sep='\t')
    bdb_assays = pd.read_csv('./DB/BindingDB/BDB_Assays.tsv', sep='\t')
    human_gpcr_pdb_info = pd.read_csv('./Input/Human_GPCR_PDB_Info.csv', sep=',')

    # Step 1: Split ENTRYID_ASSAYID into ENTRYID and ASSAYID
    bdb_rsid_eaids[['ENTRYID', 'ASSAYID']] = bdb_rsid_eaids['ENTRYID_ASSAYID'].str.split('_', expand=True)
    bdb_rsid_eaids['ENTRYID'] = bdb_rsid_eaids['ENTRYID'].astype(str)
    bdb_rsid_eaids['ASSAYID'] = bdb_rsid_eaids['ASSAYID'].astype(int)

    # Step 2: Ensure matching column types in bdb_assays
    bdb_assays['ENTRYID'] = bdb_assays['ENTRYID'].astype(str)
    bdb_assays['ASSAYID'] = bdb_assays['ASSAYID'].astype(int)

    # Step 3: Merge bdb_rsid_eaids with bdb_assays
    merged_rsid_assay = pd.merge(bdb_rsid_eaids, bdb_assays, on=['ENTRYID', 'ASSAYID'], how='left')

    # Step 4: Merge with bindingdb_all
    merged_binding_info = pd.merge(merged_rsid_assay, bindingdb_all, 
                                  left_on='REACTANT_SET_ID', right_on='BindingDB Reactant_set_id', how='left')

    # Step 5: Merge with human_gpcr_pdb_info
    human_gpcr_pdb_info['Entry'] = human_gpcr_pdb_info['Entry'].astype(str)
    final_merged = pd.merge(merged_binding_info, human_gpcr_pdb_info, 
                            left_on='UniProt (SwissProt) Primary ID of Target Chain', right_on='Entry', how='inner')

    # Step 6: Select desired columns
    final_table = final_merged[['Entry', 'Ligand SMILES', 'Ligand InChI Key', 'PubChem CID',
                                'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
                                'ASSAYID', 'DESCRIPTION']]
    final_table.to_csv('./Output/DB/BindingDB/GPCR_BindingDB_Assay.csv', index=None)
    print(f"Total rows extracted from BindingDB: {len(final_table)}")
    return final_table

# Main execution function
def main():
    """
    Execute the data parsing pipeline for DrugBank, ChEMBL, and BindingDB.
    """
    print("Starting data parsing for GPCRactDB construction...")
    
    # Parse each database
    drugbank_df = parse_drugbank()
    chembl_df = parse_chembl()
    bindingdb_df = parse_bindingdb()
    
    # Combine data (optional, depending on standardization needs)
    combined_df = pd.concat([drugbank_df, chembl_df, bindingdb_df], axis=0, ignore_index=True)
    print(f"Total combined rows: {len(combined_df)}")
    
    # Save combined data (optional, adjust path as needed)
    combined_df.to_csv('./Output/GPCRactDB_raw.csv', index=None)
    print("Data parsing completed. Files saved to ./Output/ directory.")

if __name__ == "__main__":
    main()