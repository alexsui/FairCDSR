## Create environment
All required package except for pytorch can be install using following command:
```bash
conda env create -f env.yml -n [new_env_name]
```
then you need to install pytorch independently:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Dataset folder structure (fairness_dataset)
├── **calculate_IR.ipynb**
├── **data_preprocess.py**
├── **ml_raw_data**
│   ├── mid_preprocess.ipynb
│   ├── movies.dat
│   ├── ratings.dat
│   └── users.dat
├── **Movie_lens_main**
│   ├── action_comedy
│   │   ├── Action_Comedy_overlapped.csv
│   │   ├── Action.csv
│   │   ├── Comedy.csv
│   │   ├── female_IR.json
│   │   ├── item_IF.json
│   │   ├── male_IR.json
│   │   ├── non_overlapped_Action.csv
│   │   ├── non_overlapped_Comedy.csv
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── valid.txt
│   ├── comedy_drama (same structure as action_comedy)
│   ├── drama_sci-fi (same structure as action_comedy)
│   └── sci-fi_thriller (same structure as action_comedy)
├── **RQ3_dataset**
├── **RQ4_dataset**
 **Note that RQ3_dataset and RQ4_dataset use the same folder structure as Movie_lens_main, so we omit it for brevity.**

## Data Preprocessing

This section describes the scripts used for data preprocessing and their purposes:

1. **mid_preprocess.ipynb**
   - **Purpose**: Generates domain-specific CSV files from `ml_raw_data`.
   - **Usage**: Open and run this notebook in Jupyter to output the CSV files, 
   - **Output file example**: 
       - Action.csv
       - Comedy.csv

2. **data_preprocess.py**
   - **Purpose**: Preprocesses the domain CSV files into a format suitable for model input.
   - **Usage**: Run the script from the command line:
     ```bash
     python data_preprocess.py [dataset_name]
     ```
        where `[dataset_name]` can be "Movie_lens_main", "RQ4_dataset", "RQ5_dataset"
    - **Output file example**: 
        - Action_Comedy_overlapped.csv
        - non_overlapped_Action.csv
        - non_overlapped_Comedy.csv
        - train.txt
        - valid.txt
        - test.txt
3. **calculate_IR.ipynb**
    - **Purpose**: Calculate group interaction ratio (GIR) for **Group-aware Contrastive Learning**.
   - **Usage**: Set the variable `main_dataset` as the name of the current dataset to generate GIR for the specific dataset.
   - **Output file example**: 
       - female_IR.json
       - male_IR.json



## Item Generator Pre-training

Follow the steps below to pre-train the item generator model. The pre-trained model will be stored in the `generator_model` folder.

### Running the Pre-training Script

Execute the following command from your terminal, substituting `[dataset]` with the actual name of dataset (Movie_lens_main（for RQ1~3）, RQ4_dataset, RQ5_dataset):

```
python run_generator.py [dataset] 
```


## Model Training


### Training the Recommendation Model

To train the model and obtain the recommendation results, execute the command below:

```
python train_rec.py
```
For **optimized results**, you need to locate the optimal hyperparameters stored in the RQ3 folder within the specific scenario folder. The contents in this folder will show results similar to this:
`best_params :{'alpha': 0.4, 'topk_cluster': 5, 'num_clusters_option': '300,300,300', 'substitute_ratio': 0.7, 'lambda_0': 0.8, 'lambda_1': 1.0}`
then you can set these hyperparameters in `config` folder and run `python train_rec.py` to get the results, which will be stored in the `saved_models` folder, inside a subfolder named with the `id`.
### Obtaining Results for Specific Research Questions(still required training)

For detailed results associated with different research questions, follow the guidelines provided for each RQ:
* **For RQ1, RQ2, RQ4, and RQ5:**
Run the following command, replacing `[num]` with the research question number (e.g., 1, 2, 4, or 5) and `[mode]` with the experiment mode. This command will execute the appropriate experiment script for the specified research question and mode.
 **General command format:**
    ```
    python RQ[num]/[mode]/run.py
    ```
    **Example:**
    ```
    python RQ5/substitute_mode/run.py
    ```
   
* **For RQ3:**
 To obtain results for RQ3, execute the command below. This script performs parameter tuning to optimize the model settings:
    ```
    python RQ3/param_tuning.py
    ```
    the results will be stored in the `saved_models` folder, inside a subfolder named with the `id`.