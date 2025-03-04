# convexAdam

News:
* :zap: Our [Journal paper extension](https://ieeexplore.ieee.org/abstract/document/10681158) got accepted to IEEE TMI! :tada:
* :zap: Easy installation with pip available!

## :zap: Fast and accurate optimisation for registration with little learning
![MethodOverview](images/method_overview.png?raw=true "Selfconfiguring")


## :star: ConvexAdam ranks first for the [Learn2Reg Challenge](https://learn2reg.grand-challenge.org/) Datasets! :star:

![MethodOverview](images/l2r_m.png?raw=true "Learn2RegResults")

## :floppy_disk: Installation

You can run ConvexAdam out of the box with
```
pip install convexAdam
````

## :bar_chart: Self-configuring hyperparameter optimisation

![ConceptOverview](images/sc_graphic2-2.png?raw=true "Selfconfiguring")

To obtain an automatic estimate of the best choice of all various hyperparameter configurations, we propose a rank-based multi-metric two-stage search mechanism that leverages the fast dual optimisation employed in ConvexAdam to rapidly evaluate hundreds of settings. 

We consider two scenarios: with and without available automatic semantic segmentation features using a pre-trained nnUNet. In the latter case we employ the handcraft MIND-SSC feature descriptor. For the former all infered train/test segmentations for the Learn2Reg tasks can be obtained at https://cloud.imi.uni-luebeck.de/s/cgXJfjDZNNgKRZe 

Next we create a small config file for a new task that is similar to the Learn2Reg dataset.json and contains information on which training/validation pairs to use and how many (if any) labels are available for test/evaluation. 

The entire self-configuring hyperparameter optimisation can usually be run in 1 hour or less and comprises two scripts that are executed after another.

``convex_run_withconfig.py`` and ``adam_run_with_config.py`` 

Each will test various settings, run online validation on the training/validation data and create a small log of all obtained scores that are ranked across those individual settings using a simplified version of Learn2Reg's evaluation (normalised per metric ranking w/o statistical significance and a geometric mean across metrics).

Finally you can use infer_convexadam.py to apply the best parameter setting to the test data and refer to https://github.com/MDL-UzL/L2R/tree/main/evaluation for the official evaluation.

## :books: Citations

If you find our work helpful, please cite:

```
%convexAdam + Hyperparameter Optimisation TMI
@article{siebert2024convexadam,
  title={ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration},
  author={Siebert, Hanna and Gro{\ss}br{\"o}hmer, Christoph and Hansen, Lasse and Heinrich, Mattias P},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
% Original Learn2Reg2021 Submission
@inproceedings{siebert2021fast,
  title={Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021},
  author={Siebert, Hanna and Hansen, Lasse and Heinrich, Mattias P},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={174--179},
  year={2021},
  organization={Springer}
}
% Registration with Convex Optimisation
@inproceedings{heinrich2014non,
  title={Non-parametric discrete registration with convex optimisation},
  author={Heinrich, Mattias P and Papie{\.z}, Bartlomiej W and Schnabel, Julia A and Handels, Heinz},
  booktitle={Biomedical Image Registration: 6th International Workshop, WBIR 2014, London, UK, July 7-8, 2014. Proceedings 6},
  pages={51--61},
  year={2014},
  organization={Springer}
}
```
