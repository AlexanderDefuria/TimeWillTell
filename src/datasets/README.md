


# Datasets

## Structure
data/
|- .cache 
|- datasets/ 
    # This is where all the datasets are
    |- bigvul.json
    |- ...
|- processed/
    |- {dataset}/
        |- alpaca/
        |- raw_function/
        |- graph/
configs/
|- datasets.yaml
|- models.yaml


## Processing
The pytorch Datasets are individually responsible for preprocessing the data.
`DataModule.setup()` is called by the main function to initiate the preprocessing.
It checks to see if the data is already processed. If it is then skip.

## Configs


