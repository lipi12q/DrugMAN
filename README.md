Introduction
====
        This repository contains the PyTorch implementation of DrugMAN framework, as described in out paper ‘Drug-target interaction prediction by integrating heterogeneous 
        information with mutual attention network’. DrugMAN is a modeling framework that applies multi-head attention mechanisms to learn drug-target association confidence 
        and predict drug-target interactions.DrugMAN contains two main networks. The first network intends to learn accurate and comprehensive representations for drugs and 
        protein targets from heterogeneous drug and gene/protein networks. In our model, we use the network integration algorithm BIONIC (Biological Network Integration using 
        Convolutions), which outperforms the existing state-of-the-art network embedding methods, to obtain drug/gene representation. Here, you can load your own embeddings. 
        The second network uses multi-head attention to capture and learn association information in drug-target pairs, and obtains drug-target interaction scores through 
        a series of fully connected classification layers.
Framework
====
![]
(https://github.com/lipi12q/DrugMAN/DrugMAN_framework.jpg
) 
System Requirement
====
        The source code developed in Python 3.8 using PyTorch 2.0.0. The required python dependencies are given below. DrugMAN is supported for any standard computer and operating 
        system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.
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
        $ pip install pandas ==2.0.3
        $ pip install joblib ==1.3.1
        $ pip install matplotlib ==3.7.1
        IF your computer has GPU, install the dependency package as follows:
        $ pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
        $ pip install scikit-learn == 1.3.0
        $ pip install numpy == 1.25
        $ pip install pandas ==2.0.3
        $ pip install joblib ==1.3.1
        $ pip install matplotlib ==3.7.1
Datasets
====
        The data folder contains four folders and a csv file (`all_bind.csv`). The four foders are `drug_network`, `target_network`, 
        `bionic_emb` and `warm_start` separately.






        
