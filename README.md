# Running multi-lingual training and inference pipelines

## Setup and dependencies

### Conda environment set-up

```
conda env create -f environment.yaml
```

#### Activate

```
conda activate MultiLingualEmbeddings
```

### Download  and unzip LASER

```
curl -L -O https://github.com/facebookresearch/LASER/archive/master.zip
unzip master.zip
rm master.zip
```

#### Set up LASER

```
export LASER='/Users/kevin.keenan/Developer/data-science/local-multi-lingual/LASER-master'
cd LASER_master
./install_models.sh
./install_external_tools.sh
```

##### If mecab doesn't automatically install

```
cd tools-external
curl -L -O https://github.com/taku910/mecab/archive/master.zip
unzip master.zip
cp -r <unzipped mecab dir>/mecab .
rm -r <unzipped mecab dir>
cd mecab
./configure
make
make check
make install
```

## Running the notebooks

The training and running of multi-lingual models currently happens across the four notebooks within this repo. If you are starting from scratch with only data, then the order to run the notebooks is

1. `ml-data-prep.ipynb`
2. `ml-embedding.ipynb`
3. `ml-training.ipynb`
4. `ml-inference.ipynb`

Currently, the `ml-embedding.ipynb` notebook is tailored to work with the data files generated in the data prep notebook, while a simplified embedding process is used within the inference notebook. Future work will aim to standardise the embedding process across training and inference.