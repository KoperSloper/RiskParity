import pickle
import os

def load_dataset(filepath):
    """
    Loads a pickled dataset from the specified path.
    Args:
        filepath (str): Path to the .pkl file.
    Returns:
        dict: The loaded data dictionary.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded dataset from {filepath}")
    
    # Print out the metadata to confirm what we loaded
    if 'meta' in data:
        meta = data['meta']
        print(f" -> Assets   : {meta.get('asset_cols', 'Unknown')}")
        print(f" -> Horizon  : {meta.get('horizon', 'Unknown')} months")
        print(f" -> Lookback : {meta.get('lookback', 'Unknown')} months")
    
    return data

def save_dataset(data, filepath):
    """
    Saves a dictionary to a pickle file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {filepath}")