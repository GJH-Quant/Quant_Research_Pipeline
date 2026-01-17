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
0           0.1771
1           0.3522
2           0.0354
3           0.4353

State characteristics:
        inactive_rate    avg_volume  mean_log_hl
states                                          
0            0.000360   4336.882222     0.001645
1            0.015768    1459.57366     0.000431
2            0.000301  11630.776407     0.004140
3            0.001760   2787.300034     0.000837

------------------------------------------------------------
Log(H/L) Regime Transition Matrix
------------------------------------------------------------
to         0       1       2       3
from                                
0.0   0.9451  0.0000  0.0074  0.0475
1.0   0.0000  0.9708  0.0000  0.0292
2.0   0.1126  0.0000  0.8874  0.0000
3.0   0.0146  0.0239  0.0003  0.9612

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LOG(H/L) REGIME — TRAIN vs TEST STABILITY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TV(state proportions):                 0.0713 (0 best)
Mean(|Δ mean log_hl| by state): 0.000022
Transition matrix mean abs diff:       0.0167

Per-state drift:
         p_train    p_test  mean_log_hl_train  mean_log_hl_test
states                                                         
0       0.190476  0.146470           0.001651          0.001625
1       0.330739  0.401651           0.000426          0.000442
2       0.043620  0.016333           0.004145          0.004107
3       0.435165  0.435546           0.000835          0.000842

============================================================
TOD NORMALIZED VOLUME REGIME
============================================================

State proportions:
        proportion
states            
0           0.4876
1           0.4389
2           0.0041
3           0.0694

State characteristics:
        inactive_rate    avg_volume  mean_volume_tod_z
states                                                
0            0.012329   1334.009361          -0.681481
1            0.000873   3074.936736           1.100882
2            0.000000  53535.981723          31.883767
3            0.000000   9918.035249           7.054806

------------------------------------------------------------
TOD Normalized Volume Regime Transition Matrix
------------------------------------------------------------
to         0       1       2       3
from                                
0.0   0.9065  0.0876  0.0000  0.0058
1.0   0.0962  0.8498  0.0016  0.0524
2.0   0.4293  0.1832  0.0000  0.3874
3.0   0.0264  0.3451  0.0190  0.6095

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TOD-Z VOLUME REGIME — TRAIN vs TEST STABILITY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TV(state proportions):                 0.0374 (0 best)
Mean(|Δ mean volume_tod_z| by state): 0.908630
Transition matrix mean abs diff:       0.0179

Per-state drift:
         p_train    p_test  mean_volume_tod_z_train  mean_volume_tod_z_test
states                                                                     
0       0.498657  0.462030                -0.714824               -0.598664
1       0.427732  0.464735                 1.095385                1.112527
2       0.004304  0.003548                32.652292               29.737986
3       0.069307  0.069687                 7.233263                6.646352