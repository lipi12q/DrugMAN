Introduction
====
        This repository contains the PyTorch implementation of DrugMAN framework, as described in out paper ‘Drug-target interaction prediction by integrating heterogeneous 
        information with mutual attention network’. DrugMAN is a modeling framework that applies multi-head attention mechanisms to learn drug-target interaction information 
        and Predict the probability of drug target binding.DrugMAN contains two main networks. The first network intends to learn accurate and comprehensive representations 
        for drugs and protein targets from heterogeneous drug and gene/protein networks. In our model, we use the network integration algorithm BIONIC (Biological Network Integration using 
        Convolutions), which outperforms the existing state-of-the-art network embedding methods, to obtain drug/gene representation. Here, you can also load your own embeddings. 
        The second network uses multi-head attention mechanisms to capture and learn association information in drug-target pairs, and obtains drug-target interaction scores through 
        a series of fully connected classification layers.
Framework
====
![image](https://github.com/lipi12q/DrugMAN/blob/main/%7FDrugMAN_framework.jpg) 
System Requirement
====
        The source code developed in Python 3.8 using PyTorch 2.0.0. The required python dependencies are given below. DrugMAN is supported for 
        any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.
        torch >=2.0.0
        scikit-learn >= 1.3.0
        numpy >= 1.25
        pandas >= 2.0.3
        joblib >= 1.3.1
        matplotlib >= 3.7.1
Installation Guide
====
        Firt set up a new conda environment
        # create a new conda environment
        $ conda create –name drugman pythonn==3.8
        $ conda activate drugman
        Second install required dependency packages. If your computer only has CPU, install the dependency package as follows:
        $ pip install scikit-learn == 1.3.0
        $ pip install numpy == 1.25
        $ pip install pandas == 2.0.3
        $ pip install joblib == 1.3.1
        $ pip install matplotlib == 3.7.1
        IF your computer has GPU, install the dependency package as follows:
        $ pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
        $ pip install scikit-learn == 1.3.0
        $ pip install numpy == 1.25
        $ pip install pandas == 2.0.3
        $ pip install joblib == 1.3.1
        $ pip install matplotlib == 3.7.1
Datasets
====
>The data folder contains four folders and a csv file (`all_bind.csv`). The four foders are `drug_network`,`target_network`, `bionic_emb` and `warm_start`, separately.

>`all_bind.csv`:  Drug-target interaction data are collected from five public sources including, Drugbank, map of Molecular Targets of Approved drugs (MTA), CTD, ChEMBL and BindingDB.

> `drug_network`: Four drug-related similarity networks. `Drug_disease.csv`: the disease-based drug association network. `drug_side effect.csv`: the side effect-based drug network. `drug_transciptome.csv`: the transcriptome-based drug similarity
>network.`drug_smiles.csv`: the drug structure similarity network.

>`target_network`: seven target-related similarity networks. `gene_disease.csv`: the disease-based gene association network.
>`gene_pathway.csv`: the pathway-based gene network. `gene_chromosomal.csv`:the chromosomal location-based gene network.
>`gene_transcriptome.csv`:the transcriptome-based gene similarity network. `gene_coexpression.csv`:the gene co-expression network.
>`protein_sequence.csv`: the protein sequence similarity network.

>`bionic_embed`: Embedding of drugs and targets extracted by the heterogeneous network integration algorithm BIONIC, ie `drug_features.tsv` and `target_features.tsv`. 

>`warm_start`: Five datasets obtained by `run data_split.py`. each datase is randomly divided into training, validation and test sets with a 7:1:2 ratio.

Run DrugMAN on Our Data to Reproduce Result
====
        To train DrugMAN, where we provide the whole warm-start data in `data/warm_start/`. Run the `main.py`, you start train the model. 
        The `main.py` takes about 15 minutes on GPU ram=24GB and approximately 6 hours on CPU ram=8GB. For running DrugMAN on the full warm-start data,
        we advise GPU ram>=8GB and CPU ram >=16GB. The run results will be saved in the `result` foder. 






        
