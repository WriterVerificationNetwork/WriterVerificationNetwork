# Writer Verification Network


## Installing
```shell
cd /path/to/projects
git clone git@github.com:WriterVerificationNetwork/WriterVerificationNetwork.git
cd WriterVerificationNetwork

pip install -r requirements.txt
```


## Training your own model
### Prepare data
```shell
cd /path/to/datasets
wget https://drive.switch.ch/index.php/s/zXcd2r50H4CbRzH/download?path=%2F&files=bt1_by_letters_20210824.zip
unzip bt1_by_letters_20210824.zip


cd /path/to/projects/WriterVerificationNetwork

# Generating binarization ground truth dataset 
python3 dataset/create_ground_truth.py
# A popop appears show the original and binarized image, press "0", "1" or "2" if the binarzed image is good, average or bad compared to the original image. 
# Then the binarized images are saved into the according dataset folders.
# In this project, only good binarized images are used to train the model.

# Generating image pairs from the same author


# Generating image pairs from the diffent authors

```

### Training


### Testing
