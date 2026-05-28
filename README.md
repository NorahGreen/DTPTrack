# DTPTrack
This is the official repository for **CVPR 2026** [Drift-Resilient Temporal Priors for Visual Tracking](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.pdf
).


## Prerequisites
### Environment
Assuming you have a working python environment with pip installed.

#### system packages (ubuntu)
```shell
apt update
apt install -y libturbojpeg
```
#### install pytorch
Can be skipped if using NGC container. PyTorch version should be >= 2.0.
```shell
pip install torch torchvision
```
#### extra python packages
```shell
pip install -r requirements.txt
```
This codebase should also work on Windows and macOS for debugging purposes.
### Dataset
#### Download
- [LaSOT](https://huggingface.co/datasets/l-lt/LaSOT)
- [LaSOT_Extension](https://huggingface.co/datasets/l-lt/LaSOT-ext)
- [GOT-10k](http://got-10k.aitestunion.com/downloads)
- [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit)
- [COCO 2017](https://cocodataset.org/#download)
- [TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit)
#### Unzip
The paths should be organized as follows:
```
LaSOT
├── airplane
├── basketball
...
├── training_set.txt
└── testing_set.txt

LaSOT_Extension
├── atv
├── badminton
...
└── wingsuit

GOT-10k
├── train
│   ├── GOT-10k_Train_000001
│   ...
├── val
│   ├── GOT-10k_Val_000001
│   ...
└── test
    ├── GOT-10k_Test_000001
    ...
    
TrackingNet
├── TEST
├── TRAIN_0
...
└── TRAIN_11

COCO
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000000025.jpg
    │   ...
    └── val2017
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        ...
TNL2K_TEST
├── advSamp_Baseball_game_002-Done
├── advSamp_Baseball_video_01-Done
...
```
#### Prepare ```consts.yaml```
Copy ```consts.template.yaml``` as ```consts.yaml``` and fill in the paths.
```yaml
LaSOT_PATH: '/path/to/lasot'
LaSOT_Extension_PATH: '/path/to/lasot_ext'
GOT10k_PATH: '/path/to/got10k'
TrackingNet_PATH: '/path/to/trackingnet'
COCO_2017_PATH: '/path/to/coco2017'
TNL2K_TEST_PATH: '/path/to/tnl2k_test'
```
#### Login to wandb (optional)
Register an account at [wandb](https://wandb.ai/), then login with the command:
```shell
wandb login
```
## Training & Evaluation

Note: Our code performs evaluation automatically when model training is complete.

- **Model weight** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.safetensors```.
- **Performance metrics** can be found on terminal output and wandb dashboard.
- **Tracking results** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.

### Using run.sh helper script (Linux with NVIDIA GPU only)
```shell
# Train and evaluate DTPTrack-Base model on all GPUs
./run.sh DTPTrack Base --output_dir /path/to/output
# Train and evaluate DTPTrack-Large model on all GPUs
./run.sh DTPTrack Large --output_dir /path/to/output
# Train and evaluate DTPTrack-Base model following GOT-10k protocol on all GPUs
./run.sh DTPTrack Base --output_dir /path/to/output --mixin got10k
# Train and evaluate on specific GPUs
./run.sh DTPTrack Base --output_dir /path/to/output --device_ids 0,1,2,3
# Train and evaluate on multiple nodes
./run.sh DTPTrack BBase --output_dir /path/to/output --nnodes $num_nodes --node_rank $node_rank --master_address $master_node_ip --date 2024.03.07-04.59.08-976343
```
You can set the default settings, e.g. `output_dir`, in ```run.sh```.
### Call main.py directly
```shell
# Train and evaluate DTPTrack-Base model on single GPU
python main.py DTPTrack Base --output_dir /path/to/output

# Train and evaluate DTPTrack-Base model on CPU
python main.py DTPTrack Base --output_dir /path/to/output --device cpu

# Train and evaluate DTPTrack-Base model on all GPUs
python main.py DTPTrack Base --distributed_nproc_per_node $num_gpus --distributed_do_spawn_workers --output_dir /path/to/output

# Train and evaluate DTPTrack-Base model on multiple nodes, run_id need to be set manually
python main.py DTPTrack Base --master_address $master_address --distributed_node_rank $node_rank distributed_nnodes $num_nodes --distributed_nproc_per_node $num_gpus --distributed_do_spawn_workers --output_dir /path/to/output --run_id $run_id
```
See ```python main.py --help``` for more options.

Note: If you encounter any issues with torch.compile, disable is with ```--mixin disable_torch_compile```.

Note: You can disable wandb logging with ```--disable_wandb```.

### Evaluation
Our code performs evaluation automatically when model training is complete. You can run evaluation only with the following command:
```shell
# evaluation only, on all datasets, defined in config/_dataset/test.yaml
./run.sh DTPTrack Base --output_dir /path/to/output --mixin evaluation --weight_path /path/to/weight.bin
```
The evaluated datasets are defined in ```config/_dataset/test.yaml```.

Note that, as defined in ```config/DTPTrack/run.yaml```, we evaluate LaSOT Extension dataset three times. The final performance is the average of the three evaluations.

Results are saved in ```/path/to/output/run_id/eval/epoch_{last}/```, where `run_id` is the current run ID, and `epoch_{last}` is the last epoch.

You can use the analysis scripts in pytracking derived codebase to re-calculate the metrics. [OSTrack](https://github.com/botaoye/OSTrack) is recommended. Recent SOT codebases have slightly different metrics implementations compared with earlier ones, e.g. this line in [Stack](https://github.com/researchmm/Stark/blob/162e1d37a38f92251cb380e90ccdb59608f2df3c/lib/test/analysis/extract_results.py#L77) v.s. [Pytracking](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/analysis/extract_results.py#L78).

#### TrackingNet evaluation
Once the full evaluation is done, result files are saved in ```/path/to/output/run_id/eval/epoch_{last}/TrackingNet-test.zip```.

Submit this file to the [TrackingNet evaluation server](https://tracking-net.org/) to get the result of TrackingNet test split.

### Train and evaluate with GOT-10k dataset
```shell
# Train and evaluate DTPTrack-Base model following GOT-10k protocol on all GPUs
./run.sh DTPTrack Base --output_dir /path/to/output --mixin got10k
```
Submit ```/path/to/output/run_id/eval/epoch_{last}/GOT10k-test.zip``` to the [GOT-10k evaluation server](http://got-10k.aitestunion.com/) to get the result of GOT-10k test split.

Evaluation only:
```shell
# evaluation only, on GOT-10k dataset
./run.sh DTPTrack Base --output_dir /path/to/output --mixin got10k --mixin evaluation --weight_path /path/to/weight.bin
```

Note that, as defined in ```config/DTPTrack/_mixin/got10k.yaml```, we evaluate GOT-10k dataset three times.

## VOT toolkit integration
### Install VOT toolkit
```shell
pip install vot-toolkit
```
### Download VOT dataset
prepare the VOT dataset by running the following command:
```shell
cd /path/to/vot_workspace
vot initialize vot_stack(vots2024/main|tests/multiobject)
```
fill the path to the VOT dataset in ```consts.yaml```
```yaml
VOTS2023_PATH: '/path/to/vots2023_workspace/sequences'
VOT_TESTS_MULTIOBJECT_PATH: '/path/to/vot_tests_workspace/sequences'
```
### Run VOT experiments
```shell
# Run VOT experiment (vots2024/main stack) on DTPTrack-Base with SAM-H segmentation model
python vot_main.py vots2024/main DTPTrack Large /path/to/output --mixin segmentify_sam_h --tracker_name DTPTrack  --weight_path /path/to/model_weight.bin
```

## Custom Dataset
[This page](DATASET.md) describes how to create a custom dataset for training and evaluation.

## Resumable Checkpointing
Add ```--mixin resumable``` to the command line to enable resumable checkpointing. This allows you to resume training from the last saved checkpoint if the training process is interrupted.

```shell
./run.sh DTPTrack Base --output_dir /path/to/output --mixin resumable
```

Or you can set the default value in ```run.yaml``` to:
```yaml
checkpoint:
  - type: "regular"
    epoch_trigger:
      interval: 10
      last: true
    resumable: true # false --> true
    max_to_keep: 5
```
Now the training process will save checkpoints every 10 epochs, and the last checkpoint will be saved as `recovery.yaml` in the `checkpoint` directory.

Load the last checkpoint by specifying the `--resume` argument:

```shell
./run.sh DTPTrack B-224 --output_dir /path/to/output --mixin resumable --resume /path/to/output/run_id/checkpoint/recovery.yaml
```
## Citation
```bibtex
@inproceedings{dtptrack,
    author    = {Huang, Yuqing and Lin, Liting and Zhuang, Weijun and He, Zhenyu and Li, Xin},
    title     = {Drift-Resilient Temporal Priors for Visual Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {6847-6856}
}
