from urllib.parse import quote
from urllib.request import urlopen
import ipywidgets as widgets
from IPython.display import display
from functools import partial
import numpy as np
import joblib
from tdc.single_pred import ADME
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import AllChem
from jazzy.api import molecular_vector_from_smiles as mol_vect

def CIRconvert():
    # convert molecule name written by user into SMILES
    mol_name = input("Zadejte název sloučeniny:")
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(mol_name) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        print(f"Řetězec SMILES byl úspěšně získán pro sloučeninu {mol_name}")
        return ans
    except:
        print(f'Něco se nepovedlo. Jste si jistí, jste zadali název sloučeniny "{mol_name}" správně?')
        
def on_button_click(selected, dropdowns, _):
    # Handle the button click event
    selected.clear()
    for dropdown in dropdowns:
        selected.append(dropdown.value)
    return selected
        
def inference_dataset_selection():
    # Create and display dropdown menus
    model_identifiers = ["linear", "KRR", "GB", "RF", "ANN"]
    tdc_benchmarks = ["obach", "microsome", "hepatocyte"]
    feature_types = ["morgan", "jazzy"]

    dropdowns = []

    for options, description in zip([feature_types, tdc_benchmarks, model_identifiers],
                                    ['Features:', 'Dataset:', 'Model:']):
        dropdown = widgets.Dropdown(
            options=options,
            value=options[0],
            description=description
        )
        dropdowns.append(dropdown)
        display(dropdown)

    # Create a button to trigger the action
    selected = []  # Define an empty list to store selected options
    button = widgets.Button(description="Parse Output")
    button.on_click(partial(on_button_click, selected, dropdowns))
    display(button)

    return selected  # Return the selected options list

def contains_nan(lst):
    for item in lst:
        if isinstance(item, dict):
            for v in item.values():
                if isinstance(v, (int, float)) and np.isnan(v):
                    return True
                elif isinstance(v, np.ndarray) and np.isnan(v).any():
                    return True
        elif isinstance(item, np.ndarray) and np.isnan(item).any():
            return True
    return False

def fp_from_smiles(list_smiles):
    # converts a list of SMILES strings into a list of Morgan fingerprint bit arrays

    list_fingerprint = []
    for smi in list_smiles:
        mol = Chem.MolFromSmiles(smi)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=124)
        vector = np.array(fingerprint)
        list_fingerprint.append(vector)
    return list_fingerprint

def mol_fts(smiles):
    features = []
    for smi in smiles:
        try:
            features.append(mol_vect(smi))
        except:
            # "except JazzyError" gives NameError: name 'JazzyError' is not defined
            features.append(np.nan)
    return features

def inference_predict(selected, user_smi):
    _type, benchmark, model_id = selected
    print("Zvolené parametry")
    print(f"""Látka: {user_smi}
Typ features: {_type}
Benchmark: {benchmark}
Model: {model_id}\n""")

    model_path = f"project_resources/optuna/{_type}/{benchmark}/{model_id}.joblib"
    model = joblib.load(model_path)
    dataset_names = {"obach": 'Half_Life_Obach', "microsome": 'Clearance_Microsome_AZ', "hepatocyte": 'Clearance_Hepatocyte_AZ'}
    dataset_units = {"obach": "h", "microsome": "ml.min-1.g-1", "hepatocyte": "μl.min-1.(10^6 buněk)-1"}

    if _type == "morgan":
        user_fp = fp_from_smiles([user_smi])
        y_predict = model.predict(user_fp)
    else:
        user_fp = np.array([list(fts.values()) for fts in mol_fts([user_smi])])
        y_predict = model.predict(user_fp)
        
        # if the user selected jazzy features, the predicted value needs to be inverse transformed
        # since the pretrained model predicts using scaled data
        adme_dataset = ADME(name=dataset_names[benchmark])
        adme_split = adme_dataset.get_split()
        adme_train_y = adme_split["train"]["Y"]
        adme_test_y = adme_split["test"]["Y"]
        adme_train_test_y = list(adme_train_y) + list(adme_test_y)
        reshaped_halflife = np.array(adme_train_test_y).reshape(-1, 1)
        scaler = MinMaxScaler().fit(reshaped_halflife)
        reshaped_predict = np.array(y_predict).reshape(-1, 1)
    
        y_predict = scaler.inverse_transform(reshaped_predict)[0]
    
    if not contains_nan(user_fp):
        y_predict_rounded = np.round(np.abs(y_predict)[0], decimals=1)
        benchmark_units = dataset_units[benchmark]
        if benchmark == "obach":
            print(f"Predikovaná hodnota eliminačního poločasu: {y_predict_rounded} {benchmark_units}")
        else:
            print(f"Predikovaná hodnota clearance: {y_predict_rounded} {benchmark_units}")
    else:
        print("X_test obsahuje hodnotu NaN")
