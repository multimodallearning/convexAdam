# convexAdam

## Self-configuring hyperparameter optimisation
Journal paper extension currently under review

![ConceptOverview](images/sc_graphic2-2.png?raw=true "Selfconfiguring")

To obtain an automatic estimate of the best choice of all various hyperparameter configurations, we propose a rank-based multi-metric two-stage search mechanism that leverages the fast dual optimisation employed in ConvexAdam to rapidly evaluate hundreds of settings. 

We consider two scenarios: with and without available automatic semantic segmentation features using a pre-trained nnUNet. In the latter case we employ the handcraft MIND-SSC feature descriptor. For the former all infered train/test segmentations for the Learn2Reg tasks can be obtained at https://cloud.imi.uni-luebeck.de/s/cgXJfjDZNNgKRZe 

Next we create a small config file for a new task that is similar to the Learn2Reg dataset.json and contains information on which training/validation pairs to use and how many (if any) labels are available for test/evaluation. 

The entire self-configuring hyperparameter optimisation can usually be run in 1 hour or less and comprises two scripts that are executed after another.

``convex_run_withconfig.py`` and ``adam_run_with_config.py`` 

Each will test various settings, run online validation on the training/validation data and create a small log of all obtained scores that are ranked across those individual settings using a simplified version of Learn2Reg's evaluation (normalised per metric ranking w/o statistical significance and a geometric mean across metrics).

Finally you can use infer_convexadam.py to apply the best parameter setting to the test data and refer to https://github.com/MDL-UzL/L2R/tree/main/evaluation for the official evaluation.


Learn2Reg 2021 Submission
## Fast and accurate optimisation for registration with little learning

![Slide1](images/L2R_2021_ConvexAdam.002.jpeg?raw=true "Coupled Convex")
![Slide2](images/L2R_2021_ConvexAdam.003.jpeg?raw=true "Coupled Convex")
![Slide3](images/L2R_2021_ConvexAdam.004.jpeg?raw=true "Coupled Convex")

Please see details in our paper and if you use the code, please cite the following:
Siebert, H., Hansen, L., Heinrich, M.P. (2022). Fast 3D Registration with Accurate Optimisation and Little Learning for Learn2Reg 2021. In: Aubreville, M., Zimmerer, D., Heinrich, M. (eds) Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis. MICCAI 2021. Lecture Notes in Computer Science(), vol 13166. Springer, Cham. https://doi.org/10.1007/978-3-030-97281-3_25

and

Heinrich, M.P., Papież, B.W., Schnabel, J.A., Handels, H. (2014). Non-parametric Discrete Registration with Convex Optimisation. In: Ourselin, S., Modat, M. (eds) Biomedical Image Registration. WBIR 2014. Lecture Notes in Computer Science, vol 8545. Springer, Cham. https://doi.org/10.1007/978-3-319-08554-8_6



## Excellent results on Learn2Reg 2021 challenge
- for multimodal CT/MR registration (Task1) 
- intra-patient lung CT alignment (Task2)
- and inter-patient whole brain MRI deformations (Task3)
[Challenge Website](https://learn2reg.grand-challenge.org)

![Slide4](images/L2R_2021_ConvexAdam.005.jpeg?raw=true "Coupled Convex")

![Results](images/l2r2021_convexAdam.png?raw=true "Results")
