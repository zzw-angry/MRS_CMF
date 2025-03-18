This repository is provided to offer the code for the framework MRS_CMF. 
The required data is not included in this repository, but it can be downloaded from here: (https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG). 
To execute the necessary code, first run the command：
python -u build_iib_graph.py --dataset=baby --topk=2 
Then, execute the training command: 
python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
