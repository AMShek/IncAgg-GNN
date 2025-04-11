# ReverbGNN

This repository provides the code, data, and results for the paper: *ReverbGNN: Scaling Graph Training with Periodical Local and Global Memory*

<p float="left">
  <img src="./figures/overview.png" width="100%" />
  <!-- <img src="./figures/instr-108.jpg" width="54%" />  -->
</p>


## Requirements

This project is built upon [Python 3.10](https://www.python.org).

PyG library is required for this project, please install according to your pytorch and CUDA version.
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

where `${TORCH}` should be replaced by either `1.7.0` or `1.8.0`, and `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, `cu110` or `cu111`, depending on your PyTorch installation.

For a complete list of required packages, please find them in the `requirements.txt` file.
It is recommended to create a new `conda` environment for this project as it may be tricky to install PyQt as it can mess up your current dependencies.

```bash
conda create -n rung python=3.10
conda activate rung

pip install -r requirements.txt
```


## Reproducing Results

### Train the model
```bash
python clean.py --model='RUNG' --norm='MCP' --gamma=6.0 --data='cora'
```

### PGD attack on the trained model
```bash
python attack.py --model='RUNG' --norm='MCP' --gamma=6.0 --data='cora'
```


## Experimental Results



<p float="left">
  <img src="./figures/results.png" width="100%" />
  <!-- <img src="./figures/instr-108.jpg" width="54%" />  -->
</p>


