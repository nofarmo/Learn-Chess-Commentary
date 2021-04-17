# Learn Chess Commentary

## Description
Given a game, such as chess, there is a lot of data that given a position in the game, a commentator can explains the current position. We collected a dataset from an online chess forum (340k samples) and added more chess features, in order to train a chess commentator.

## Training

To train the model see **train_script.py**, you can change parameters in **Configs/** folder.

Our dataset attached at this [Drive folder](https://drive.google.com/drive/folders/1b-HxT47mZ2V7ex7rv0lH-ut2ka0I92kF?usp=sharing)

It was collected from the online chess website www.gameknot.com, and with addional info using python-chess, sample contains:

![Sample](https://github.com/nofarmordehai/Learn-Chess-Commentary/blob/main/Sample%20example.jpg)

## Checkpoints

Our pretrained checkpoint (for GPT2 and BERT-BASE) attached at this [Drive folder](https://drive.google.com/drive/folders/17_I9gUQNY_QP4hBNikPI0klUH5chS-ML?usp=sharing)

## Inference

Try **test_notebook.ipynb** to load checkpoints, examine results and calculate metrics by yourself.
