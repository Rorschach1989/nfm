# Neural frailty machine

This is a minimal implementation of NFM framework in the paper ``Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions``

The python module ``nfm`` contains all functionalities of nfm. For demo purpose, we provide two scripts over the metabric dataset (included in the ``data`` repository)
run
```shell
python experiment_metabric_pf.py
```
or 
```shell
python experiment_metabric_fn.py
```
Note that the preprocessing routine in the paper is slightly different from the standard routine in ``pycox`` package