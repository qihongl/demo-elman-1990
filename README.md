# demo-elman-1990

here's a qualitatively replication of [1] -- by the end of training, the model's prediction error peaks right after word/event boundaries. the design of the experiment is inspired by [2]. 


Play with the model: pytorch <a href="https://colab.research.google.com/github/qihongl/demo-elman-1990/blob/master/elman_pytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory">
</a>; 
psyneulink <a href="https://colab.research.google.com/github/qihongl/demo-elman-1990/blob/master/elman_pnl.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory">
</a>
- these two versions are not meant to be quantitatively matched 


<img src="https://github.com/qihongl/demo-elman-1990/blob/master/imgs/pe.png" alt="PE over time">


<br>
References: 

[1] Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211. https://doi.org/10.1016/0364-0213(90)90002-E

[2] Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. Science, 274(5294), 1926–1928. https://doi.org/10.1126/science.274.5294.1926
