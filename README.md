# metabolism-hyperparameter-optimization
Uživatel může predikovat vlastnosti libovolné látky pomocí předtrénovaných modelů skrze [interaktivní inference MetHPO](https://colab.research.google.com/drive/1xaqE8QNvFGhT2YMnqG9va2hSS4X8MTca?usp=sharing).

Datasety z [Therapeutics Data Commons](https://tdcommons.ai/) využívané pro trénování a testování modelů: [Obach et al.](https://tdcommons.ai/single_pred_tasks/adme/#half-life-obach-et-al), [AstraZeneca](https://tdcommons.ai/single_pred_tasks/adme/#clearance-astrazeneca).

**Jazzy-data-prep.ipynb** → Generování molekulárních deskriptorů pomocí [Jazzy](https://github.com/AstraZeneca/jazzy) a spolu s half-life/clearance ukládání do csv souborů do project_resources/jazzy_splits.\
**mols-visualization.ipynb** → 2D i 3D vizualizace některých molekul, pro správné fungování 3D modelů je potřeba stáhnout [Jupyterlab_3dmol](https://github.com/3dmol/jupyterlab_3Dmol) extention.\
**TDC-ADME.ipynb** → Strojové učení pomocí modelů ze [Scikit-learn](https://scikit-learn.org/stable/) na ECFP a molekulárních deskriptorech.\
**ADME-data-analysis.ipynb** → Zpracovávání výsledků strojového učení na benchmarcích TDC-ADME, generování grafů.\
**inference-model-prep.ipynb** → Příprava a ukládání modelů pro inference; tyto modely jsou trénovány jak na trénovacích, tak testovacích datech z TDC-ADME.\
**interactive-inference.ipynb** → Interaktivní inference - predikování vlastností libovolné látky z jejího názvu pomocí zvoleného předtrénovaného modelu.

## project_resources
**jazzy_splits/** → csv soubory data splitů s molekulárními deskriptory.

## docking_resources
**2ij7.pdb** → Krystalografická struktura enzymu CYP121 s ligandem. [zdroj](https://www.rcsb.org/structure/2IJ7)\
**2ij7_edited.{pdb, pdbqt}** → Pouze jeden řetězec bez ligandů a bez molekul vody.\
**2ij7_docked.pdb** → 2ij7_edited.pdb s dockovaným ligand_docked.pdb.\
**2ij7_Compound_1.pdb** → CYP121 s aktivovanou formou kyslíku na hemové skupině.\
**2ij7_*.cxs** → ChimeraX session, pomocí kterých jsem získával obrázky dockingu.\
**ligand.{mol, pdb, pdbqt}** → 3,6-bis[(4-hydroxyphenyl)methyl]piperazine-2,5-dione; molekula 1 z obrázku 1 z [tohoto článku](https://pubmed.ncbi.nlm.nih.gov/23620594/).\
**ligand_docked.{pdb, pdbqt}** → Dockované ligandy vygenerované pomocí AutoDock Vina.
