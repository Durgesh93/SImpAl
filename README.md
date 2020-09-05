# Your Classifier can Secretly Suffice Multi-Source Domain Adaptation

Code for the NeurIPS 2020 submission (ID #10432).

## Setup

Ensure that you have python2.7 and all the required dependencies installed.

	pip install -r requirements.txt

Note, for the GUI interface (image retrieval demo), the TKinter package is required. Install it using:

	sudo apt-get install python-tk


This project is organized into 4 folders ```code```, ```data```, ```summaries```, ```weights```

	msda_upload/
	├── code
	│   ├── img_ret_code
	│   ├── inference_code
	│   ├── train_code
	│   ├── pad_code
	│   └── union_disjoint_code
	├── data
	│   ├── domain-net
	│   ├── image-clef
	│   ├── office-31
	│   ├── office-caltech
	│   ├── office-home
	│   └── README.md
	├── summaries
	├── weights
	│   └── README.md
	├── image_retrieval_demo.mp4
	├── README.md (this file)
	├── Supplementary.pdf
	└── requirements.txt

### Hardware requirements

This project requires an NVIDIA-GPU with CUDA 9.0 to run PyTorch with GPU computing capabilities. The GPU ID can be defined in ```gpu_id``` in ```config.py``` file under respective code folder.


## Dataset

Please refer to README.md file in ```msda_upload/data``` folder for setting up the data for expermients.


## Code

There are 5 folders in code folder 
	- ```train_code```            	- contains script for training
	- ```inference_code```        	- contains inference script
	- ```union_disjoint_code```   	- contains script for category-shift (overlap, disjoint) experiment (see Sec. 4.2)
	- ```img_ret_code```          	- contains code for image search application (see supplementary)
	- ```pad_code```				- contains scripts for calculating proxy-A distance (see Fig. 4b)     


Each code folder contains a ```config.py``` file where settings related to each experiments can be changed. Before execution, change the ```server\_root\_path```  ,```gpu``` avariable in ```config.py``` file.


### Training the model (```train_code```)
	
Model training script is written in ```train_code``` folder. Before running the training script, the experiment related settings can be changed in ```config.py``` file.

Some import settings related to the experiments are as follows:

- ```dataset_name``` - dataset for the experiment (the possible values are "office-caltech", "office-31", "office-home", "image-clef" and "domain-net")
- ```data_key```	 - task configuration for the dataset (e.g. for office-caltech dataset, the possible configurations are "ACD_W", "ADW_C", "ACW_D", "CDW_A")
- ```id_str```       - a unique id string to be appended at the end of each experiment name
- ```gpu_id```		 - the ID of the GPU on which the code must be run
- ```tb_port_no```   - tensorboard port number (by default it is set to 9999 - gpu_id)

All possible values for ```dataset_name``` and ```data_key``` can be referred to in ```config_populate.py``` file which is imported by ```config.py```. The variable ```expt_dict``` contains settings for ```enough_iter```, ```max_iter```. Here, ```enough_iter``` refers to the number of iterations for performing warm-start, and ```max_iter``` refers to the total number of training iterations. (See Algorithm 1 in the paper).

After making the relevant changes in ```config.py``` file, execute the following commands in order:
		
	python create_index_list.py 
	python train_supervised.py

The file ```create_index_list.py``` creates the data splits for each dataset, while ```train_supervised.py``` starts the training process.

The training related metrics can be tracked in tensorboard. By default, tensorboard will start in port number indicated by ```tb_port_no``` in ```config.py```.

- Target agreement rate (Fig. 5a, bottom) is tracked by the key ```classifier_agreement_metric```
- Pseudo-label accuracy (Fig. 5b) for agreement and disagreement region is tracked by ```hf_acc_g_precision_agree``` metric, and ```hf_acc_g_precision_disagre``` metric respectively
- Migration of target samples with correct pseudo-labels (Fig. 6a) from agreement to disagreement region is shown by ```hf_acc_f_agree_recall``` (fraction of correctly pseudo-labeled target samples in agreement region) and ```hf_acc_f_dis_agree_recall``` metrics respectively.
- Accuracy of samples ranked based on w (Fig. 6b) is tracked by ```acc_conf_plot``` under the Images tab in Tensorboard.
- Target accuracy (Fig. 5a)  is tracked  by ```cls_acc``` (or ```cls_acc_data```) metrics

These metrics are defined in metrics.py file


### Inference of the pre-trained model (```inference_code```)

Inference of a pre-trained model can done using the scripts in inference_code folder. Before running the inference sciript, the pre-trained model weights can be downloaded from: https://drive.google.com/drive/folders/1Y_8L2tEq6oPIpJqxEsSZNIuQOai_qfI8?usp=sharing

Each model is stored in ```.pth``` file in the folder named using the experiment name with which the model was trained. For eg. an experiment run on domain-net dataset with configuration CIPQS->R has the weights folder named as:

	expt_resnet101_domain-net_CIPQS_R_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1
	├── expt_data.pth
	├── model_120000.pth
	├── model_444000.pth
	├── model_enough_iter120000.pth
	├── opt_120000.pth
	├── opt_444000.pth
	└── opt_enough_iter120000.pth

Different configuration settings related to experiment is stored in ```config.py``` file. One can change ```dataset_name```, ```data_key```, ```settings['bb']``` (backbone) variables for selecting different models. The trained weights will be mapped from ```weights_dict``` and the corresponding model will be selected automatically.

Please note that we report the accuracy in the paper corresponding to the model after the last training iteration, although this code base saves weights for the best performing model.

To run the inference script, ensure that ```config.py``` is set up, and execute the following command:

	python inference.py


### Running experiments in the category-shift settings (```union_disjoint_code```)

The model can be trained under the category-shift settings (See 4.2b) as follows. First, change the shared classes and source private classes under Overlap and Disjoint settings in the ```config_populate.py``` file. Then, make the relevant changes in ```config.py``` file (same as training the model). Then, execute the following commands in order:

	python create_index_list.py 
	python train_supervised.py


### Cross domain image retrieval tool (```img_ret_code```)

Cross domain image retrieval for domain-net dataset can be done using this tool. To run the tool, download the pre-trained domain-net model weights from: https://drive.google.com/drive/folders/1Y_8L2tEq6oPIpJqxEsSZNIuQOai_qfI8?usp=sharing.

After downloading the weights, set the folder name ```src_datasets```, ```trgt_dataset``` in ```config.py``` file. To run the tool execute the following command,

	python img_retrieval.py

Please see image_retrieval_demo.mp4 for a demonstration. 

### Code for calculating Proxy-A(PAD) distance (```pad_code```)

To run code for PAD distance execute the following command

	python pad_dist.py


### tSNE Plots

The tSNE plots can be obtained using the standard sklearn library.
