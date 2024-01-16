Introduction
====
    This repository contains the PyTorch implementation of DrugMAN framework, as described in out paper ‘Drug-target interaction prediction by integrating heterogeneous 
    information with mutual attention network’. DrugMAN is a modeling framework that applies multi-head attention mechanisms to learn drug-target association confidence 
    and predict drug-target interactions.DrugMAN contains two main networks. The first network intends to learn accurate and comprehensive representations for drugs and 
    protein targets from heterogeneous drug and gene/protein networks. In our model, we use the network integration algorithm BIONIC (Biological Network Integration 
    using Convolutions), which outperforms the existing state-of-the-art network embedding methods, to obtain drug/gene representation. Here, you can load your own embeddings. 
    The second network uses multi-head attention to capture and learn association information in drug-target pairs, and obtains drug-target interaction scores through a series of
    fully connected classification layers.
Framework
====
![]
(https://github.com/lipi12q/DrugMAN/.gif) 
