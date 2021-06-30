# OAK_module
OAK module that groups all the scripts and functions necessary for neonatal infant monitoring.

## Enviroment configuration
Execute with conda installed:
```
conda env create -f env.yml
conda activate oak
pip install -r requirements.txt
```

If new libraries are installed, it is necessary to execute the following command:
```
pip-compile --extra=dev
```