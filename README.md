GENERAL INFORMATION
-----------------------------

Dataset Title:

Python Code and Supporting Data for Analysis of Cascaded Online Learning Flight Control System With Smoothened Actions

Author: Yifei Li
              Section Control & Simulation
              Faculty of Aerospace Engineering
              Delft Universty of Technology

Corresponding author: Yifei Li

Contact Information: y.li-34@tudelft.nl

Delft University of Technology
Faculty of Aerospace Engineering
Kluyverweg 1, 2629 HS, Delft 
The Netherlands


DESCRIPTION
-----------
General description of the dataset: 

The dataset contains relevant Python code and supporting data in relation to CHAPTER 3 of the dissertation 'Safe and Sample Efficient Reinforcement Learning for Flight Control' by Yifei Li (2025). It concerns the effects of action smoothness techniques on the cascaded online learning flight control system. The PyTorch package serves as the main instrument for simulation. For more information, we refer to the respective thesis chapter/pulication.

[Keywords: Flight Control, Reinforcemen Learning, Cascaded structure, Temporal Smoothness, Filter]


ACCESS INFORMATION
------------------

All data files (e.g., .txt) and code scripts (e.g., .py files) are licensed under the 3-Clause BSD liscense (https://opensource.org/license/bsd-3-clause).

Copyright notice:
Technische Universiteit Delft hereby disclaims all copyright interest in the code scripts presented in this dataset. All code scripts have been written by the Author.

Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.
© 2025, Yifei Li

Dataset DOI: 10.4121/22a294e7-3550-4e6a-b415-12107e3f159c


VERSIONING AND PROVENANCE
-------------------------

Last modification date (YYYY-MM-DD): 2025-05-13


METHODOLOGICAL INFORMATION
--------------------------

Description of data collection methods:

The main data were generated using Numpy and PyTorch packages for model and algorithm simulation.
        
Instrument- or software- specific information used by the author to interpret the data:

    1. Ubuntu 20.04.5 LTS (x64)
    2. Anaconda3 (x64)
    3. Pycharm 2022.3 (Community Edition)


FILE OVERVIEW
-------------

Directory structure:
1)Code -Main Python scripts; see _README in subdirectory for more details. The aerial vehicle model is a reproduction of the nonlinear simulation model described in the paper by R.A. Hull, D. Schumacher, and Z. Qu. Design and Evaluation of Robust Nonlinear Missile Autopilots from a Performance Perspective. In: Proceedings of American Control Conference, pp. 189–193, 1995.
2)Data_vanilla -.txt data files for figure reproduction of the vanilla control system.
3)Data_TS -.txt data files for figure reproduction of the temporal smooth control system.
4)Data_TS&Filter -.txt data files for figure reproduction of the temporal smooth&filter control system.

REFERENCES
----------

No extra references used for data collection/generation, processing and visualization

