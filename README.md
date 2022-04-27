# RSD
RSD (RNA-seq deconvolutioner) is an automatic protocol for RNA-Seq deconvolution and obtaining CD8 T cell specific sc RNA-seq.


## environment
RSD mainly dependents on tensorflow2. For the environment, 
we strongly recommend using conda:
* Install [conda](https://docs.conda.io/projects/conda/en/latest/).
* Create environment:<br>
`conda env create -f RSD.yml -n RSD`
* Enter environment:<br>
`conda activate RSD`

## usage
RSD is a UNIX style command line tool. For usage, type in:<br>
`python3 main.py -h`