Introduction
====
> This repository contains the PyTorch implementation of DrugMAN framework, as described in out paper ‘Drug-target interaction prediction by integrating heterogeneous 
> information with mutual attention network’. DrugMAN is a modeling framework that applies multi-head attention mechanisms to learn drug-target interaction information and
> Predict the probability of drug target binding. DrugMAN contains two main networks. The first network intends to learn accurate and comprehensive representations
> for drugs and protein targets from heterogeneous drug and gene/protein networks. In our model, we use the network integration algorithm [BIONIC](https://github.com/bowang-lab/BIONIC),
> which outperforms the existing state-of-the-art network embedding methods, to obtain drug and gene representation. Here, you can also load your own embeddings. 
> The second network uses multi-head attention mechanisms to capture and learn association information in drug-target pairs, and obtains drug-target interaction scores through 
> a series of fully connected classification layers.

Framework
====
![image](https://github.com/lipi12q/DrugMAN/blob/main/%7FDrugMAN_framework.jpg) 

System Requirement
====
> The source code developed in Python 3.8 using PyTorch 2.0.0. The required python dependencies are given below.
> DrugMAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run.
> There is no additional non-standard hardware requirements.
                
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
> `data/drug_network`: Drug-related similarity networks. <br>
> `data/gene_network`: Gene-related similarity networks. <br>
> `data/all_bind.csv`: Drug-target interaction data. <br>
> `drug_emb.json`: Input file of the heterogeneous network algorithm BIONIC to extracted drug represtation.<br>
> `target_emb.json`: Input file of the heterogeneous network algorithm BIONIC to extracted target represtation.<br>
> we alse provide the demo dataset in `data/warm_start` and the drug/target feature trained by Bionic in `data/bionic_emb` to reproduce the training process.<br>
Run DrugMAN on Our Data to Reproduce Result
====
> There are four steps to complete the process of training the model. <br>
> *Firt, run `data_split.py` to obtain train, validation and test set data. <br>
> *Second, run `drug_emb.json` using BIONIC algorithm to extracted drug represtation.<br>
> *Third, run `target_emb.json` using BIONIC algorithm to extracted target represation.<br>
> *Forth, run `main.py`. note that you need to create a `result` foder to store computational result.<br>
> we alse provide the `demo.ipynb`, A more detailed training procedure is available here.<br> 
> The `main.py` takes about 15 minutes on GPU ram=24GB and approximately 6 hours on CPU ram=8GB. For running DrugMAN,
> we advise GPU ram>=8GB and CPU ram >=16GB.





        
