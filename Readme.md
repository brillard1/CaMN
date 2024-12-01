# Cross-aligned Fusion For Multimodal Understanding

This is an official repository of Cross-aligned Multimodal Network (CaMN) framework.

## CaMN pipeline

### Environment setup
Install dependencies from `env.yml` file using the below command:

```
conda env create -f env.yml
```

### Data Preparation
- Download the dataset by running `setup.sh`

- We use Stanford CoreNLP (version 3.9.2) lemmatizing, POS tagging, etc.

- Generate AMR for language dataset using:
    ```
    python amr_generation.py {fin.txt} {fout.txt}
    ```
    fin is train, dev or test and fout is the output file

    Or use the stog model from [here](https://github.com/sheng-z/stog) to generate it

- Download the glove embeddings by following the process given [here](https://github.com/stanfordnlp/GloVe)

### Model
To train or evaluate the model, execute the following script:

```
sh run.sh
```

- Specify the task (task1, task2_merged, task3) and mode (train, eval) in it
- Number of epochs, batch size and device parameters can be specified here
- See args.py for the exact arguments parsed