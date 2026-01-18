[SANITY] bars/day summary (filled):
count    241.0
mean     390.0
std        0.0
min      390.0
25%      390.0
50%      390.0
75%      390.0
max      390.0
dtype: float64
[SANITY] OK: no bars outside RTH
[SANITY] inactive rate: 0.64%
[SANITY] fraction(high==low): 1.51%

============================================================
LOG(H/L) VOLATILITY REGIME
============================================================

State proportions:
        proportion
states            
0           0.5366
1           0.3842
2           0.0792

State characteristics:
        inactive_rate   avg_volume  mean_log_hl
states                                         
0            0.011222  1737.930905     0.000514
1            0.000914  3510.726799     0.001132
2            0.000269  7897.994223     0.003076

------------------------------------------------------------
Log(H/L) Regime Transition Matrix
------------------------------------------------------------
to         0       1       2
from                        
0.0   0.9783  0.0217  0.0000
1.0   0.0322  0.9586  0.0092
2.0   0.0000  0.0757  0.9243

============================================================
TOD NORMALIZED VOLUME REGIME
============================================================

State proportions:
        proportion
states            
0           0.7792
1           0.0085
2           0.2123

State characteristics:
        inactive_rate    avg_volume  mean_volume_tod_z
states                                                
0            0.008111   1766.669953          -0.216818
1            0.000000  36583.421973          23.686073
2            0.000351   5739.499298           3.474924

------------------------------------------------------------
TOD Normalized Volume Regime Transition Matrix
------------------------------------------------------------
to         0       1       2
from                        
0.0   0.9462  0.0007  0.0532
1.0   0.3615  0.0000  0.6385
2.0   0.1870  0.0259  0.7871