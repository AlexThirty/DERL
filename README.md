# Derivative Learning (DERL)
Code for the paper **[Learning and Transferring Physical Models through Derivatives](https://openreview.net/forum?id=IbBCDDeDF7)** accepted at Transactions on Machine Learning Research (TMLR)

Please consider citing us

	@article{
        trenta2026derl,
        title={Learning and Transferring Physical Models through Derivatives},
        author={Alessandro Trenta and Andrea Cossu and Davide Bacciu},
        journal={Submitted to Transactions on Machine Learning Research},
        year={2026},
        url={https://openreview.net/forum?id=IbBCDDeDF7},
    }


## Requirements


1. Create the environment
    - ``` python -m venv derl ```
    or  ``` conda create --name derl python=3.12 ```

2. Activate the environment

3. Install the required packages
    - ``` pip install requirements.txt ```

## Run the experiment
Each folder represents a single experiment. To run an experiment, enter its folder and follow these steps:
1. Generate the data

    ```python generate.py```

2. Run the experiment

    ```python train.py --mode Derivative --device [DEVICE]```
    - ```mode``` represents the model used. ```Derivative``` is for DERL, ```Output``` is OUTL, Sobolev is SOB and ```Output+PINN``` is for OUTL+PINN.
    - In general, look at the file to see the particular arguments to be provided, such as the ```name``` one, which is used to distinguish between grid data and random data.
    - The models automatically use the tuned parameters available in ```tuner_results.py```. You can run tuning with Ray with the ```tune.py``` files.

3. Test and plot

    ```python test_joint.py```