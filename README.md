### Local Similarity Pattern and Cost Self-Reassembling for Deep Stereo Matching Networks  

<img src="figure/figure.png" width="60%" height="50%">

### Contributions
- A novel pairwise feature LSP to extract structural information, which is beneficial for accurate matching especially when the illumination of the image pair is imbalanced
- A novel disparity refinement method CSR (or DSR to save memory) to deal with outliers that are difficult to match, e.g. disparity discontinuities and occluded regions.

#### Dependencies：
- Python 3.6
- PyTorch 1.7.0
- torchvision 0.3.0
- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [KITTI stereo 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)


#### Training on SceneFlow

```bash
python train.py --data_path (your Scene Flow data folder)
```

#### Finetuning on KITTI

```bash
python KITTI_ft.py --data_path (your KITTI training data folder) --load_path (the path of the model trained on SceneFlow)
```
