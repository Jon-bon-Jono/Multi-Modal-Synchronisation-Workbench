Tested 2 tasks using 2D poses:

1. Joint locations in the image: average positions and spread of each joint across the people present
2. Body shape task: based on root-relative 2D pose normalized wrt to body size (divided by distance between shoulder and hip midpoint), then predict the same average positions and spread of each joint across people present. 
   
Model predicts the per-joint mean and standard deviation of all ground truths present. This summary avoids needing identity association. 

Trained two models using the `piecewise_rgb_to_pc_v001_map` mapping, one processed individual points and another smaller network processed radar statistics. Compared each network with a simple predictor that always just returned the per-joint average ground truth. 
* Body-shape error was 2% higher than fixed average for the model which processed individual points and 10% lower for statistics model
* Joint-location error was 54% lower for the model which processed individual points and 62% lower for statistics model

Overall, joint-location task performed much better than body shape task


* Shifting training poses by 0.2s did not consistently increase error. 
* Using roughly half the anchors did not consistently increase error. 
* Shifting poses by 10s worsened joint-location prediction, but did not worsen body-shape prediction overall. 


How is this a proxy task for pose estimation? This is just pose estimation? Didn't the `synchronisation_validation_hpe_proxy.md` file suggest a task involving trying to match radar frames to a specific pose ground truth using a fixed window size?


Raw 19_MM anchored by Peter Pan
Raw 09_SY being anchored by Chris Kim

