# Dataset Prepare
## Training (KITTI Dataset & Extended Data)
The preparation of the KITTI dataset follows [Monodepth2](https://github.com/nianticlabs/monodepth2/tree/master#-kitti-training-data).

### Download

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P your_kitti_dataset_save_path/
```
Then unzip with
```shell
cd your_kitti_dataset_save_path
unzip "*.zip"
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to `unzip`!


**For ease of use, we recommend using symbolic links to prepare the dataset, e.g.**

```shell
ln -s your_kitti_dataset_save_path ./data/kitti
```

### Splits

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.

### Extended Data

Note that we extend the KITTI dataset with random conruption types and severities. **Executing the following command will create different extended data each time:**
```shell
python create_aug_pairs.py
```

## Evalation (KITTI-C Dataset) 

The preparation of the KITTI-C dataset follows [RoboDepth](https://github.com/ldkong1205/RoboDepth/blob/main/docs/DATA_PREPARE.md#kitti-c).

### Download

The corrupted KITTI test sets under Eigen split can be downloaded from Google Drive with [this](https://drive.google.com/file/d/1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa/view?usp=sharing) link.

Alternatively, you can directly download them to the server by running:
```shell
mkdir your_kitti_c_dataset_save_path
cd your_kitti_c_dataset_save_path
```

```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa" -O kitti_c.zip && rm -rf /tmp/cookies.txt
```
Then unzip with:
```shell
unzip kitti_c.zip
```
**Warning:** This dataset weighs about **12GB**, make sure you have enough space to `unzip` too!

**For ease of use, we recommend using symbolic links to prepare the dataset, e.g.**

```shell
ln -s your_kitti_c_dataset_save_path/kitti_c ./data/kitti_c
```

### Splits

The KITTI-C dataset shares the same split files with the KITTI dataset.

For `eigen` splits, please follow the [download_gt.md](../splits/eigen/download_gt.md) to download depth gt data.