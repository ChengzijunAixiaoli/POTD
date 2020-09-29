
# PPMM

Python3 and R implementation of the paper [Sufficient dimension reduction for classification using principal optimal transport direction] (NeurIPS 2020)

Principal optimal transport direction (POTD) is a sufficient dimension reduction (SDR) method, which utilizes the principal displacement vectors of the optimal transport between the data respecting different response catogories to form the basis of the SDR subspace. Different from existing SDR methods, POTD is powerful for identifying the SDR subspace for catogorical response data, especially for binary response data. For the Python3 implementation, Sinkhorn algorithm could be used for fast calculation, while for the R implementation, the computation may takes longer time.

Feel free to ask if any question.

If you use this toolbox in your research and find it useful, please cite PPMM using the following bibtex reference:

```
@incollection{meng2020potd,
title = {Sufficient dimension reduction for classification using principal optimal transport direction},
author = {{Meng}, Cheng and {Yu}, Jun and {Zhang}, Jingyi and
 {Ma}, Ping, and {Zhong}, Wenxuan},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020}
}
```

### Prerequisites
* Python (>= 3.6)
* Numpy (>= 1.11)
* Matplotlib (>= 1.5)
* For Optimal transport [Python Optimal Transport](https://pot.readthedocs.io/en/stable/) POT (>=0.5.1)


### What is included ?

* Python3 and R implementation of POTD

* Demo notebooks:
	- [potd_example.ipynb](potd_example.ipynb): Compare POTD with other methods in estimating the SDR subspace
### Authors

* [Cheng Meng](https://github.com/ChengzijunAixiaoli)




## References

[1] Flamary Rémi and Courty Nicolas [POT Python Optimal Transport library](https://github.com/rflamary/POT)
