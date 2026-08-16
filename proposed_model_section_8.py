
 =======================================================================================================================================
DEPTH-OT V2 — UNCERTAINTY-GATED ARGMAX × CLUSTERING HYBRID
=======================================================================================================================================
Theta                : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_independent_depth_confidence_dev_search/a08_l010_g000_dev_theta.npy
Records              : /content/drive/MyDrive/depth_ot_patent/data/processed/dev_records.pkl
Top-50 predictions   : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_theta_space_clustering_dev_search/top50_theta_space_predictions.npz
Clustering ranking   : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_theta_space_clustering_dev_search/theta_space_clustering_dev_ranking.csv
Output               : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search
Uncertainty types    : ['entropy', 'margin', 'max_probability', 'combined_rank']
Selection modes      : ['all', 'disagreement']
Replacement fractions: 200
Split                : DEV ONLY
=======================================================================================================================================

[THETA]
Shape          : (9855, 30)
Minimum        : 9.9999941659e-09
Maximum        : 9.9999971000e-01
Row-sum minimum: 1.000000000000
Row-sum maximum: 1.000000000000

[CPC LABEL COUNTS]
section : 9
class   : 123
subclass: 484

[BASE VALIDATION]
Mean Pur_p  : 0.369728
Mean Pur_a  : 0.427735
Mean NMI    : 0.307660
Section NMI : 0.224564
Class NMI   : 0.323990
Subclass NMI: 0.374425
[PASS] Base predictions reproduced.

[UNCERTAINTY STATISTICS]
entropy          | min=0.000002 | mean=0.150930 | max=0.617860
margin           | min=0.000000 | mean=0.316835 | max=0.999865
max_probability  | min=0.000000 | mean=0.187390 | max=0.774610
combined_rank    | min=0.000101 | mean=0.500051 | max=0.998647

[CANDIDATE PREDICTIONS]
Unique candidate arrays: 51

[ALIGNMENT SUMMARY]
top50::pow300_pca25_gmm_tied_s42                                            | agreement=0.9892 | disagreement=0.0108
top50::pow400_pca25_spherical_kmeans_s42                                    | agreement=0.9891 | disagreement=0.0109
top50::pow400_pca25_spherical_kmeans_s73                                    | agreement=0.9889 | disagreement=0.0111
top50::pow400_pcanone_spherical_kmeans_s73                                  | agreement=0.9886 | disagreement=0.0114
top50::pow400_pca25_gmm_tied_s42                                            | agreement=0.9884 | disagreement=0.0116
top50::pow400_pcanone_spherical_kmeans_s42                                  | agreement=0.9884 | disagreement=0.0116
top50::pow400_pcanone_spherical_kmeans_s17                                  | agreement=0.9883 | disagreement=0.0117
top50::pow400_pca25_spherical_kmeans_s17                                    | agreement=0.9870 | disagreement=0.0130
top50::pow300_pca25_spherical_kmeans_s17                                    | agreement=0.9850 | disagreement=0.0150
top50::pow300_pcanone_spherical_kmeans_s73                                  | agreement=0.9848 | disagreement=0.0152
top50::consensus_top15                                                      | agreement=0.9842 | disagreement=0.0158
top50::pow300_pcanone_spherical_kmeans_s17                                  | agreement=0.9841 | disagreement=0.0159
top50::pow300_pca25_spherical_kmeans_s73                                    | agreement=0.9836 | disagreement=0.0164
top50::consensus_top30                                                      | agreement=0.9833 | disagreement=0.0167
top50::pow400_pca20_spherical_kmeans_s42                                    | agreement=0.9833 | disagreement=0.0167
top50::pow300_pca25_spherical_kmeans_s42                                    | agreement=0.9832 | disagreement=0.0168
top50::consensus_top20                                                      | agreement=0.9832 | disagreement=0.0168
top50::pow400_pca20_spherical_kmeans_s17                                    | agreement=0.9830 | disagreement=0.0170
top50::pow200_pca25_gmm_tied_s42                                            | agreement=0.9824 | disagreement=0.0176
top50::pow200_pcanone_spherical_kmeans_s17                                  | agreement=0.9802 | disagreement=0.0198

[HYBRID SEARCH PLAN]
Candidate arrays   : 51
Uncertainty types : 4
Selection modes   : 2
Fractions         : 200
Estimated hybrids : 81,600
Hybrid DEV search: 100% 81600/81600 [11:38<00:00, 116.71it/s]
[CODE SAVED] /content/drive/MyDrive/depth_ot_code/scripts/search_uncertainty_gated_hybrid_dev.py

=========================================================================================================================================================================================
UNCERTAINTY-GATED HYBRID — DEV CPC RANKING
=========================================================================================================================================================================================
 rank                    candidate_source uncertainty_type selection_mode replacement_fraction_requested replacement_fraction_actual section_nmi class_nmi subclass_nmi mean_pur_p mean_pur_a mean_nmi delta_mean_nmi  stable_candidate
    1 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.475000                    0.047590    0.224884  0.327383     0.380582   0.372806   0.417047 0.310950       0.003290             False
    2 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.510000                    0.051040    0.224552  0.327473     0.380747   0.372840   0.415525 0.310924       0.003264             False
    3 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.130000                    0.050533    0.224717  0.327399     0.380637   0.372907   0.415728 0.310918       0.003258             False
    4 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.505000                    0.050533    0.224717  0.327399     0.380637   0.372907   0.415728 0.310918       0.003258             False
    5 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.135000                    0.051852    0.224244  0.327591     0.380919   0.372670   0.415390 0.310918       0.003258             False
    6 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.480000                    0.048097    0.224820  0.327365     0.380544   0.372738   0.416980 0.310910       0.003250             False
    7 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.515000                    0.051547    0.224370  0.327500     0.380770   0.372704   0.415424 0.310880       0.003220             False
    8 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.500000                    0.050127    0.224720  0.327312     0.380528   0.372840   0.415863 0.310854       0.003194             False
    9 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.525000                    0.052562    0.224149  0.327546     0.380844   0.372670   0.415221 0.310847       0.003187             False
   10 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.520000                    0.052055    0.224154  0.327523     0.380845   0.372569   0.415255 0.310841       0.003181             False
   11 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.485000                    0.048605    0.224678  0.327269     0.380564   0.372738   0.416574 0.310837       0.003177             False
   12 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.470000                    0.047083    0.224789  0.327227     0.380491   0.372670   0.417081 0.310836       0.003176             False
   13 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.120000                    0.047793    0.224737  0.327286     0.380470   0.372738   0.416980 0.310831       0.003171             False
   14 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.495000                    0.049619    0.224631  0.327264     0.380495   0.372772   0.415999 0.310797       0.003137             False
   15 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.490000                    0.049112    0.224658  0.327256     0.380465   0.372772   0.416269 0.310793       0.003133             False
   16 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.125000                    0.049315    0.224632  0.327266     0.380465   0.372738   0.416100 0.310787       0.003128             False
   17 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.465000                    0.046575    0.224813  0.327036     0.380288   0.372670   0.417149 0.310712       0.003052             False
   18 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.500000                    0.050127    0.224028  0.327447     0.380635   0.372806   0.416303 0.310703       0.003044             False
   19 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.140000                    0.053070    0.223839  0.327445     0.380789   0.372670   0.414950 0.310691       0.003031             False
   20 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.530000                    0.053070    0.223839  0.327445     0.380789   0.372670   0.414950 0.310691       0.003031             False
   21 top50::hellinger_pca25_gmm_tied_s42  max_probability            all                       0.115000                    0.046474    0.224792  0.327009     0.380222   0.372670   0.417149 0.310674       0.003015             False
   22 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.525000                    0.052562    0.223972  0.327309     0.380740   0.372840   0.415288 0.310674       0.003014             False
   23 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.495000                    0.049619    0.224016  0.327381     0.380609   0.372704   0.416303 0.310669       0.003009             False
   24 top50::hellinger_pca25_gmm_tied_s42    combined_rank            all                       0.145000                    0.052461    0.223972  0.327311     0.380688   0.372806   0.415322 0.310657       0.002997             False
   25 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.455000                    0.045561    0.224800  0.326967     0.380174   0.372603   0.417419 0.310647       0.002987             False
   26 top50::hellinger_pca25_gmm_tied_s42  max_probability   disagreement                       0.460000                    0.046068    0.224758  0.326928     0.380217   0.372637   0.417284 0.310634       0.002974             False
   27 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.530000                    0.053070    0.223961  0.327217     0.380725   0.372840   0.414882 0.310634       0.002974             False
   28 top50::hellinger_pca25_gmm_tied_s42    combined_rank            all                       0.135000                    0.049112    0.224072  0.327261     0.380556   0.372603   0.416607 0.310630       0.002970             False
   29 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.490000                    0.049112    0.224072  0.327261     0.380556   0.372603   0.416607 0.310630       0.002970             False
   30 top50::hellinger_pca25_gmm_tied_s42    combined_rank   disagreement                       0.520000                    0.052055    0.223910  0.327247     0.380670   0.372738   0.415390 0.310609       0.002949             False

=======================================================================================================================================
FINAL DECISION
=======================================================================================================================================
Base Mean NMI              : 0.307660

BEST RAW HYBRID
Candidate source           : top50::hellinger_pca25_gmm_tied_s42
Uncertainty                : max_probability
Selection mode             : disagreement
Requested fraction         : 0.4750
Actual replacement fraction: 0.0476
Mean Pur_p                 : 0.372806
Mean Pur_a                 : 0.417047
Mean NMI                   : 0.310950
Delta Mean NMI             : +0.003290
Section NMI                : 0.224884
Class NMI                  : 0.327383
Subclass NMI               : 0.380582

BEST STABLE HYBRID
Candidate source           : top50::hellinger_pca25_gmm_tied_s42
Uncertainty                : margin
Selection mode             : all
Requested fraction         : 0.0750
Actual replacement fraction: 0.0350
Mean Pur_p                 : 0.372332
Mean Pur_a                 : 0.420396
Mean NMI                   : 0.310544
Delta Mean NMI             : +0.002884
Section NMI                : 0.224812
Class NMI                  : 0.327480
Subclass NMI               : 0.379340

SELECTED CANDIDATE
Selected type              : stable
Selected Mean Pur_p        : 0.372332
Selected Mean Pur_a        : 0.420396
Selected Mean NMI          : 0.310544
Run TEST now               : False
Decision                   : BORDERLINE_HYBRID_CANDIDATE_REVIEW_BEFORE_TEST
Runtime                    : 11.70 minutes

OUTPUT FILES
Ranking CSV                : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/uncertainty_gated_hybrid_dev_ranking.csv
Summary JSON               : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/uncertainty_gated_hybrid_dev_summary.json
Best raw predictions       : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/best_raw_hybrid_dev_predictions.npy
Best stable predictions    : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/best_stable_hybrid_dev_predictions.npy
Selected predictions       : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/selected_hybrid_dev_predictions.npy
Selected assignments       : /content/drive/MyDrive/depth_ot_patent/results/depth_ot_v2/depth_ot_v2_patent_semantic_seed42_20260814_055110/epoch016_uncertainty_gated_hybrid_dev_search/selected_hybrid_dev_assignments.csv
Saved code                 : /content/drive/MyDrive/depth_ot_code/scripts/search_uncertainty_gated_hybrid_dev.py

[BORDERLINE] Review stability before TEST.
=======================================================================================================================================
[PASS] Uncertainty-gated hybrid DEV search completed.
