# Representational Analysis of Reber Grammar in a Simple Recurrent Network
Project summary under construction...

Until then, please see `Shih_CLPS1492-Writeup.pdf` for background information on the project and the complete report of the methods and results.

## Help with navigating this repository:
* Folders prefixed with a `+` denote Matlab script packages. The `+` makes the folder act like a namespace in Matlab. I used it for organization.
* All the neural network models used in this project are in NN_models. These models were built with the [emergent framework](http://emersim.org/) using the biologically-based [Leabra](https://en.wikipedia.org/wiki/Leabra) algorithm. In theory, you should be able to uncover similar representations by building the same architecture using classic neural networks. The key is to include enough units per layer to allow distributed representations. Details are available in the methods section of the write up.
* To investigate what the neural network model has learned, I examined the patterns of activity of each layer as the model performs the Reber grammar task. This data can be found in the data/ folder. There are data from 11 of the same models trained on the same Reber grammar task. I did this because I was not sure if the networks would all learn the same way given that the initial weights are random.
* `NNClass_Epoch_Stats.m` and `NNClass_SSE_Summary_Stats.m` are just some descriptive stats from the earlier neural network training step.
* Running `setup_files.tcsh` parses the data export from the neural network model. It has already been run on the data in the data/ folder.
* Details are in the write up but in short, I ran binary classification (one-vs-rest) to classify Reber grammar node transitions and letter transitions. For simplicity I chose to use logistic regression. I tried it with MAP (`NNClass_LogRegr_princeton.m`) and MLE (`NNClass_LogRegr_II.m`), and went with MLE because it performed better and the slower training wasn't an issue with this amount of data.
* `NNClass_Classification_Results.m` summarizes the classification results
* `NNClass_Hierarchical_Clustering.m` performs hierarchical clustering for insight on the learned representations
