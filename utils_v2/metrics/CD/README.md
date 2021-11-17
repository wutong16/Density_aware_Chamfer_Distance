# Pytorch Chamfer Distance (CD).

### What is the Chamfer Distance ? 

[Stanford course](http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf) on 3D deep Learning

Include a **CUDA** version, and a **PYTHON** version with pytorch standard operations.
NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt thresholds accordingly.

- [x] F - Score  


### Usage

```python
import torch, chamfer3D.dist_chamfer_3D, fscore
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
f_score, precision, recall = fscore.fscore(dist1, dist2)
```

### Aknowledgment 

Original backbone from [Fei Xia](https://github.com/fxia22/pointGAN/blob/master/nndistance/src/nnd_cuda.cu).

JIT cool trick from [Christian Diller](https://github.com/chrdiller)

Modify from [Thibault GROUEIX](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)


