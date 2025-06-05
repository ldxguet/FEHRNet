# FEHRNet

## This project primarily references the following repository
* https://github.com/open-mmlab/mmpose

## Environment Configuration:
* Python 3.9
* PyTorch 1.8 or above
* pycocotools (Linux: `pip install pycocotools`; Windows: `pip install pycocotools-windows` (no additional VS installation required))
* Ubuntu or CentOS (Windows not recommended)
* GPU training is highly recommended
* Detailed environment configuration:
```
Package                 Version      Editable project location
----------------------- ------------ ------------------------------------
absl-py                 2.1.0
addict                  2.4.0
aliyun-python-sdk-core  2.15.1
aliyun-python-sdk-kms   2.16.3
apex                    0.1
attrs                   23.2.0
audioread               3.0.1
av                      12.1.0
beautifulsoup4          4.12.3
Brotli                  1.0.9
certifi                 2024.6.2
cffi                    1.16.0
charset-normalizer      2.0.4
chumpy                  0.70
click                   8.1.7
colorama                0.4.6
coloredlogs             15.0.1
contourpy               1.2.1
coverage                7.5.3
crcmod                  1.7
cryptography            42.0.8
cycler                  0.12.1
Cython                  3.0.10
decorator               4.4.2
decord                  0.6.0
einops                  0.8.0
exceptiongroup          1.2.1
filelock                3.14.0
flake8                  7.1.0
flatbuffers             24.3.25
fonttools               4.53.0
fsspec                  2024.6.0
gdown                   5.2.0
grpcio                  1.64.1
huggingface-hub         0.23.4
humanfriendly           10.0
idna                    3.7
imageio                 2.34.1
imageio-ffmpeg          0.5.1
imgaug                  0.4.0
imgviz                  1.7.5
importlib_metadata      7.1.0
importlib_resources     6.4.0
iniconfig               2.0.0
interrogate             1.7.0
isort                   4.3.21
jmespath                0.10.0
joblib                  1.4.2
json-tricks             3.17.3
kiwisolver              1.4.5
labelme                 5.5.0
lazy_loader             0.4
librosa                 0.10.2.post1
llvmlite                0.43.0
lmdb                    1.4.1
Markdown                3.6
markdown-it-py          3.0.0
MarkupSafe              2.1.5
matplotlib              3.9.0
mccabe                  0.7.0
mdurl                   0.1.2
mkl-fft                 1.3.8
mkl-random              1.2.4
mkl-service             2.4.0
mmcv                    2.1.0
mmcv-full               1.4.2
mmdet                   3.3.0
mmengine                0.10.4
mmpose                  1.3.1        /home/ldx/python_project/mmpose-main
model-index             0.1.11
moviepy                 1.0.3
mpmath                  1.3.0
msgpack                 1.0.8
munkres                 1.1.4
natsort                 8.4.0
networkx                3.2.1
numba                   0.60.0
numpy                   1.26.4
onnx                    1.16.1
onnxruntime             1.18.0
opencv-contrib-python   4.10.0.82
opencv-python           4.10.0.82
opendatalab             0.0.10
openmim                 0.3.10
openxlab                0.1.0
ordered-set             4.1.0
oss2                    2.17.0
packaging               24.1
pandas                  2.2.2
parameterized           0.9.0
pillow                  10.3.0
pip                     24.0
platformdirs            4.2.2
pluggy                  1.5.0
pooch                   1.8.2
proglog                 0.1.10
protobuf                4.25.3
py                      1.11.0
pycocotools             2.0.8
pycodestyle             2.12.0
pycparser               2.22
pycryptodome            3.20.0
pyflakes                3.2.0
Pygments                2.18.0
pyparsing               3.1.2
PyQt5                   5.15.11
PyQt5-Qt5               5.15.15
PyQt5_sip               12.15.0
PySocks                 1.7.1
pytest                  8.2.2
pytest-runner           6.0.1
python-dateutil         2.9.0.post0
PyTurboJPEG             1.7.3
pytz                    2023.4
PyYAML                  6.0.1
QtPy                    2.4.1
requests                2.28.2
rich                    13.4.2
safetensors             0.4.3
scikit-image            0.22.0
scikit-learn            1.5.0
scipy                   1.13.1
setuptools              60.2.0
shapely                 2.0.4
six                     1.16.0
soundfile               0.12.1
soupsieve               2.6
soxr                    0.3.7
sympy                   1.12.1
tabulate                0.9.0
tensorboard             2.17.0
tensorboard-data-server 0.7.2
termcolor               2.4.0
terminaltables          3.1.10
threadpoolctl           3.5.0
tifffile                2024.5.22
timm                    1.0.3
tomli                   2.0.1
torch                   1.13.1
torchaudio              0.13.1
torchsummary            1.5.1
torchvision             0.14.1
tqdm                    4.65.2
typing_extensions       4.11.0
tzdata                  2024.1
urllib3                 1.26.18
Werkzeug                3.0.3
wheel                   0.43.0
xdoctest                1.1.5
xtcocotools             1.14.3
yapf                    0.40.2
zipp                    3.19.2
```

## File Structure:
```
  ├── model: Code for building FEHRNet-related modules
  ├── train_utils: Training and validation related modules (including COCO validation)
  ├── my_dataset_coco.py: Custom dataset for reading COCO2017 dataset
  ├── person_keypoints.json: Information about human keypoints in COCO dataset
  ├── train.py: Single GPU/CPU training script
  ├── train_multi_GPU.py: For users utilizing multiple GPUs
  ├── predict.py: Simple prediction script using trained weights for inference
  ├── validation.py: Validate/test data COCO metrics using trained weights and generate record_mAP.txt file
  └── transforms.py: Data augmentation related functions
```

## Dataset - This example uses the COCO2017 dataset
* COCO official website: https://cocodataset.org/
* Here we take downloading the COCO2017 dataset as an example, mainly downloading three files:
    * `2017 Train images [118K/18GB]`: All image files used during training
    * `2017 Val images [5K/1GB]`: All image files used during validation
    * `2017 Train/Val annotations [241MB]`: Corresponding annotation JSON files for training and validation sets
* Extract all files to the `coco2017` folder to obtain the following directory structure:
```
├── coco2017: Dataset root directory
     ├── train2017: All training image folder (118,287 images)
     ├── val2017: All validation image folder (5,000 images)
     └── annotations: Corresponding annotation folder
              ├── instances_train2017.json: Training set annotation file for object detection and segmentation tasks
              ├── instances_val2017.json: Validation set annotation file for object detection and segmentation tasks
              ├── captions_train2017.json: Training set annotation file for image captioning
              ├── captions_val2017.json: Validation set annotation file for image captioning
              ├── person_keypoints_train2017.json: Training set annotation file for human keypoint detection
              └── person_keypoints_val2017.json: Validation set annotation file for human keypoint detection
```

## Training Method

* Ensure the dataset is prepared in advance
* Ensure proper configuration of `--num-joints` (number of keypoints for human detection, COCO uses 17 points), `--fixed-size` (height and width of input target images, default [256, 192]), and `--data-path` (pointing to the `coco2017` directory)
* For training, directly use the train.py training script
