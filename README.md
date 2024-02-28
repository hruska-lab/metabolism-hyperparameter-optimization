# metabolism-hyperparameter-optimization
**Jazzy-data-prep.ipynb** → Generování molekulárních deskriptorů pomocí [Jazzy](https://github.com/AstraZeneca/jazzy) a spolu s half-life/clearance ukládání do csv souborů do project_resources/jazzy_splits.\
**mols-visualization.ipynb** → 2D i 3D vizualizace některých molekul, pro správné fungování 3D modelů je potřeba stáhnout [jupyterlab_3dmol](https://github.com/3dmol/jupyterlab_3Dmol) extention.\
**TDC-ADME.ipynb** → Strojové učení pomocí modelů ze [scikit-learn](https://scikit-learn.org/stable/) na ECFP a molekulárních deskriptorech.

## project_resources
**jazzy_splits/** → csv soubory data splitů s molekulárními deskriptory.

## docking_resources
2ij7.pdb -> Krystalografická struktura enzymu CYP121 s ligandem. [zdroj](https://www.rcsb.org/structure/2IJ7)\
2ij7_edited.{pdb, pdbqt} -> Pouze jeden řetězec bez ligandů a bez molekul vody.\
2ij7_docked.pdb -> 2ij7_edited.pdb s dockovaným ligand_docked.pdb.\
2ij7_Compound_1.pdb -> CYP121 s aktivovanou formou kyslíku na hemové skupině.\
2ij7_*.cxs -> ChimeraX session, pomocí kterých jsem získával obrázky dockingu.\
ligand.{mol, pdb, pdbqt} -> 3,6-bis[(4-hydroxyphenyl)methyl]piperazine-2,5-dione; molekula 1 z obrázku 1 z [tohoto článku](https://pubmed.ncbi.nlm.nih.gov/23620594/).\
ligand_docked.{pdb, pdbqt} -> Dockované ligandy vygenerované pomocí AutoDock Vina.
