# VideoVirtualTryon-HCAI-Project
Applying virtual cloth tryon on videos. Human Centered AI course project

### setup
```bash
git clone https://github.com/manishmanu/VideoVirtualTryon-HCAI-Project.git
git submodule init && git submodule update
mkdir ckp
```

Download FlowNet checkpoint from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing). \
Download FlowStyle checkpoints from [here](https://drive.google.com/drive/folders/1hunG-84GOSq-qviJRvkXeSMFgnItOTTU?usp=sharing). \
Place both the checkpoints in `ckp` folder created above.

Note: FlowStyle checkpoints should have both `aug` and `non_aug` versions. You can chose one while running the test.py (as shown below).

### Run
```bash
python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 1 --warp_checkpoint ../ckp/aug/PFAFN_warp_epoch_101.pth --gen_checkpoint ../ckp/aug/PFAFN_gen_epoch_101.pth --dataroot video_test
```
Note: \
Add `--enable_flow` to the arguments to enable optical flow while video generation. \
`video_test` folder should have a cloth, edge mask of cloth and a video of size 192 x 256.


### Result
Check the results in `results` folder, that gets created at the root of the project. `result_with_flow.avi` will be generated if flow is enabled and `result_without_flow.avi` will be generated if flow is not enabled.


