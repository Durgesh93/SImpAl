The experiment can be done on 5 data sets office-31, office-home, office-caltech,image-clef and domain-net.
For office-31,office-home,office-caltech and image-clef datasets, the  evaluation protocal is the same as MFSAN.
For domain-net we use evaluation protocal same as M3SDA and use train/test splits provided below.

				Instruction to create respective datasets
				-----------------------------------------

Please organize the folder as,

./dataset_name/class_name/image

For DomainNet dataset, place the train.txt and test.txt inside index_main folder, for eg. ./domain-net/index_main/clipart_train.txt


Links to download the dataset are as follows

* office-31 dataset can be downloaded here: https://drive.google.com/open?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE

* office-home dataset can be downloaded here: http://hemanthdv.org/OfficeHome-Dataset/

* image-clef dataset can be downloaded here: https://drive.google.com/drive/folders/1Y_8L2tEq6oPIpJqxEsSZNIuQOai_qfI8?usp=sharing

* office-caltech dataset can be downloaded here: https://drive.google.com/drive/folders/1Y_8L2tEq6oPIpJqxEsSZNIuQOai_qfI8?usp=sharing

* domain-net dataset (.zip contains images, and train.txt / test.txt correspond to the train and test splits)
	clipart
	http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt

	infograph
	http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt

	painting
	http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt

	quickdraw
	http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt


	real
	http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt

	sketch
	http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
	http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt
