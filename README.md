The directory is meant for setup on the hendrix cluster and can be run there as a slot in solution. 

All code use batch script to run, batch script used for the latest run of each file is kept here:

/Luna16 contains the data directly taken from the Luna16 website.
/luna16_cls3d contains the output from the classifier. 

/exp_utils contains data setup for use in experiments.

/experiments is the out directory for every experiment. 

/attrs contains nem_utils and the baseline implementations for other explaination models. 
/nem_utils contains the checkpoint for the latest NEM training, it also contains the main NEM implementation, model extractors and method. 

The rest of the files are in the upper directory.

An example of running the code on the cluster is:

Go to NEM3D directory.

run:
sbatch run_train_nem.sbatch. 

The directory also contains the code for the luna16 classifier. This code is setup for google collab environment. 

