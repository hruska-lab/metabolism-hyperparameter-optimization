# datacytochromy
**data-cleaning-splitting.ipynb** → formátování dat ze zdrojů (databáze [ChEMBL](https://www.ebi.ac.uk/chembl/) a [PubChem](https://pubchem.ncbi.nlm.nih.gov/)), rozdělování na train-test\
**mols-visualization.ipynb** → 2D i 3D vizualizace některých molekul, pro správné fungování 3D modelů je potřeba stáhnout [jupyterlab_3dmol](https://github.com/3dmol/jupyterlab_3Dmol) extention\
**scikit-learn.ipynb** → strojové učení pomocí modelů ze [scikit-learn](https://scikit-learn.org/stable/) na bit arrays získané ze smiles, Tanimoto podobnosti molekul\
**jazzy.ipynb** → strojové učení pomocí modelů ze scikit-learn na features vygenerované pomocí [Jazzy](https://github.com/AstraZeneca/jazzy)\
**nequip.ipynb** → příprava souborů pro využívání s [NequIP](https://github.com/mir-group/nequip)

## project_resources
**AID_1508591_datatable_all.csv, AID_1508603_datatable_all.csv** → výchozí csv pro RLM.csv [zdroj](https://pubchem.ncbi.nlm.nih.gov/bioassay/1508591) a HLC.csv [zdroj](https://pubchem.ncbi.nlm.nih.gov/bioassay/1508603)\
**ChEMBL_3A4.csv** → výchozí csv pro 3A4.csv [zdroj](https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/target_chembl_id%3ACHEMBL340) (kvůli velikosti souboru jsou vynechané standard type IC50, Inhibition, INH, TDI, Ratio IC50)\
**base_splits/** → již rozdělené csv soubory na train a test - buď náhodně (random/), nebo pomocí [DeepChem ScaffoldSplitter](https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#scaffoldsplitter) (scaffold_splitter/)\
**jazzy_splits/** → stejné jako base_splits, ale csv obsahují Jazzy features

## nequip
**minimal_eng.yaml** → příkladový yaml soubor z [GitHubu NequIPu](https://github.com/mir-group/nequip/blob/main/configs/minimal_eng.yaml) a výchozí yaml soubor pro *isozyme*_*splitter*_config.yaml\
**positions/** → soubory s daty pro strojové učení s NequIP (&#42;.extxyz) a kopie těchto souborů včetně indexů molekul (&#42;.txt)
