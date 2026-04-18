# MH4518_Group8

## Getting Started

### 1. Fetch YFinance Historical Datasets and Generating Synthetic Option Chain
Before running the main application, you must navigate to the `create_dataset` directory and run the data generation scripts from **within** that folder as __name__ == "__main__".

```bash
# Navigate into the creation directory
cd create_folder

# In order to run the following scripts below:
#Script Path 1: `MH4518_Group8/create_dataset/real_world/fetch_historical_v1.py`
#Script Path 2: `MH4518_Group8/create_dataset/theoretical_synthetic_options/blackScholes_synthetic.py`
#Script Path 3: `MH4518_Group8/create_dataset/theoretical_synthetic_options/heston_synthetic.py`

# From within their respective directories, run the 3 data generation scripts below:
python fetch_historical_v1.py
python blackScholes_synthetic.py
python heston_synthetic.py

```

Their successful execution and terminal outputs are copied and pasted into seperate .txt files in the following directories below for your reference:

- MH4518_Group8/create_dataset/real_world/output_ref.txt

- MH4518_Group8/final_dataset/synthetic_option/black_scholes_output_ref.txt

- MH4518_Group8/final_dataset/synthetic_option/heston_output_ref.txt


### 2. Models Python Reference and Respective Papers
Models Python Reference:
- MH4518_Group8/models

Repective Papers:
- MH4518_Group8/papers


### 3. Jupyter Files
Run the below jupyter files for output graphs e.g. variance reduction and backtest graph

```bash

MH4518_Group8/pipeline/any_model/cev_last.ipynb
MH4518_Group8/pipeline/any_model/cev.ipynb
pMH4518_Group8/ipeline/any_model/gbm.ipynb
MH4518_Group8/pipeline/any_model/heston.ipynb

```

### 4. Risk Assessment
Finally, you can run the below risk assessment script to understand how we obtained our risk assessment numbers and figures (for a more quantitative view)

```bash

python risk_assessment.py

```

Thank you !

