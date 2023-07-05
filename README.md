# signature-regime-detection

Code accompanying the paper "Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures", B. Horvath, Z. Issa, 2023.

Code used to run the experiments found in Section 4 onwards can be found in the <code>notebooks</code> folder. The repository is organized as follows:

1) <code>data</code>: Folder for storage of pre-calculated data assets. These include simulated paths, MMD scores, and so on.
2) <code>src</code>: Source code for the project. 
3) <code>notebooks</code>: Step-by-step guides outlining each of the experiments shown in the paper. Includes:
   1) <code>4-online-regime-detection.ipynb</code>: Subsections 4.1, 4.2, 4.3, 4.4 from the paper 
   2) <code>4-higher-rank-mmd.ipynb</code>: Subsection 4.5
   3) <code>4-detection-comparisons.ipynb</code>: Subsection 4.6
   4) <code>4-non-markovian-detection.ipynb</code>: Subsection 4.7
   5) <code>5-clustering-pathwise-regimes.ipynb</code>: Section 5
   6) <code>6-real-data-experiments.ipynb</code>: Section 6

Direct any questions to `zacharia.issa@kcl.ac.uk`.

****

All required packages can be found in <code>requirements.txt</code>. 

Some users may have issues installing the <code>higherOrderKME</code> package, which is necessary to perform signature kernel MMD calculations. 

This seems to be due to an issue with an older version of <code>h5py</code>. 

If this is the case, we recommend manually installing <code>higherOrderKME</code>, removing the <code>h5py</code> requirement from <code>setup.py</code>.