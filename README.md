# demo-elman-1990

here's a qualitatively replication of [1] -- by the end of training, the model's prediction error peaks right after word/event boundaries. the design of the experiment is inspired by [2]. 

<img src="https://github.com/qihongl/demo-elman-1990/blob/master/imgs/pe.png" alt="PE over time">


`rnn_pnl_sl.ipynb` uses <a href="https://princetonuniversity.github.io/PsyNeuLink/">psyneulink</a> (tested on the `psyneulink/devel`, 02/17/2019) and `rnn_sl.ipynb` uses pytroch.
these two versions are not meant to be quantitatively matched 


<br>
References: 

[1] Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211. https://doi.org/10.1016/0364-0213(90)90002-E
[2] Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. Science, 274(5294), 1926–1928. https://doi.org/10.1126/science.274.5294.1926
