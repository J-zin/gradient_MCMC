# gradient_MCMC
visualising gradient based MCMC methods

- [x] [Langevin dynamics](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=321EE83B91BA3766CBB02BF6ABEB5751?doi=10.1.1.226.363&rep=rep1&type=pdf)
- [x] [Stein variantional gradient decent (SVGD)](https://arxiv.org/pdf/1608.04471.pdf)

### Run

```
python runexp.py
```

### 1D Gaussian Mixture

$$
p(x) = \frac{1}{3}\mathcal{N}(-2, 1) + \frac{2}{3}\mathcal{N}(2, 1)
$$

| <img src="image/1dsvgd.gif" width="100%"> | <img src="image/1dld.gif" width="100%"> |
| :--------------------------------------: | :------------------------------------: |

<img src="image/kde.gif" width="50%">

### 2D Gaussian Mixture

$$
p(x) = \frac{5}{4}\mathcal{N}(\left[\begin{array}{c}
    5\\  
    5\\
  \end{array}\right], \left[\begin{array}{cc}
    1 & 0 \\ 
    0 & 1 \\
  \end{array}\right]) + \frac{1}{5}\mathcal{N}(\left[\begin{array}{c}
    -5\\  
    -5\\
  \end{array}\right], \left[\begin{array}{cc}
    1 & 0 \\ 
    0 & 1 \\
  \end{array}\right])
$$

| <img src="image/2dsvgd.gif" width="100%"> | <img src="image/2dld.gif" width="100%"> |
| :--------------------------------------: | :------------------------------------: |

