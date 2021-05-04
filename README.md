# fastMRI

Modifications:
Created the files row_selector.py and row_selector_improved.py to train a network for guessing rows with high likelihood of having a high sum of magnitudes in fastmri_examples/varnet
Created row_picker.py for k-space data exploration in fastmri_examples/varnet
Created VarNetMaxRowDataTransform to test effectiveness of choosing rows based on the sums of the magnitudes in fastmri/data/transforms.py
Created VarNetSmartChooseDataTransform to utilize models created in row_selector files to be used in training the VarNet in fastmri/data/transforms.py
Created OffsetMaskFunc class to test equispacing masking variant in fastmri/data/subsample.py
Slight modification of singlecoil_train_varnet_demo.py to implement all of the the above


Here is the link to my Colab implementation of FastMRI for mask optimization. It required a modified requirements.txt file as the standard requirements.txt file was incompatible with Colab 
https://colab.research.google.com/drive/19QE4beASK1r1Co_Ke-1ZBvuepuz6Hu8X?usp=sharing

Here is the link to my Colab runner for i-RIM for fastMRI
https://colab.research.google.com/drive/1kwhdxqWSuZ-vMHmx1d6ssQBr5WAn9xPE?usp=sharing
