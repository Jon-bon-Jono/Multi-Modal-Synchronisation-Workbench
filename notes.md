Dear Agents,

Please do not modify the contents of this file. Thankyou. 


Please propose a method for exporting training data from SyncWB so that it could be used for training point-cloud-based HPE models. The long-term plan would be to export the entire re-synchronised Living Lab dataset (raw point cloud only) for publication. The current short-term plan is to export subsets of the Living Lab dataset (raw point cloud) so that they could be used to train/fine-tune models in this framework: `C:\Users\GSBME\Documents\cardiac\VIP_mmWave_HPE_demo` (specifically the `vip_hpe_ml` module). We should be able to use SyncWB to export a subset consisting of all records from one or more subjects and exactly one type of mapping-version and point cloud version. We don't want it to be possible to create an export involving:
* multiple different versions of the same `DEVICE_RUN`
* two or more separate `DEVICE_RUN`s with different mapping versions/methods and/or different point cloud versions.
For instance, we don't want an export to involve raw point cloud generated offline using settings A mixed with raw point cloud generated offline using settings B and/or mixed with point cloud generated online (there should only be one point cloud version involved in each export). We also want all data in an export to be mapped using a single mapping method, we don't to mix mapping methods in the export. 

In vip_hpe_ml, we will soon build another implementation of the DatasetAdapter protocol but for the Living Lab dataset (like how `PackedMMFiAdapter` is the protocol implementation for the MM-Fi dataset) and this will be able to handle the exported SyncWB datasets. Please treat `C:\Users\GSBME\Documents\cardiac\VIP_mmWave_HPE_demo` as read-only. I would prefer not to structure the files in the export such that we have one file for each `RUN_SAMPLE` or for each mapped pair of `RUN_SAMPLE`s. It would probably be best to have one file for each combination of `DEVICE_RUN`. Your proposition should not involve any implementation, just explain what you plan to do, e.g. what file type, HDF5?




Select one processing recipe revision and one synchronization release - exactly one processing_recipe_id (related to offline processing settings), exactly one mapping_profile_id (sync algo)

So basically, when we export we will select specific subjects, mapping versions and point cloud versions, the SyncWB export tool will verify the selection can be exported (based on the constraints we had defined and the rejection criteria), and then create a processing_recipe_id and mapping_profile_id for the export? YES - the exporter would derive and compare the recipe/profile IDs before accepting the selection, rather than assign a common ID to whatever was selected.




I would prefer for each modality to be located in separate files. For example, the Kinect pose labels should not be in the same file as the point cloud. Prospective users of our dataset might be interested in different tasks like pose-based activity recognition, point-cloud-based pose estimation. So they shouldn't be expected to load all data modalities. For now, we will only export point cloud and Kinect pose. Therefore, for example, if the Kinect recording reset once while radar recording was still running, we should have 4 files in total (point cloud and pose file for the first overlapping segment, and point cloud and pose for the second overlapping segment).

In regards to the geometry, I would prefer for the radar point cloud and Kinect pose to be saved in the same way that it is displayed in the SyncWB GUI (e.g. units, frame of reference, axes, handedness, Kinect-to-radar transform). The Kinect-to-radar transform should be clearly documented, as this will change once we properly calibrate the sensors. 

I agree with the radar-centred correspondence table where each radar frame is assigned a nearest Kinect-frame time. Although, the first subset version that we will export will involve the default initial mapping between Kinect and raw radar offline point cloud frames, so we shouldn't necessarily need to have applied a specialised re-synchronized mapping to perform this export. Also, some of the pose estimation models that we will be training do require (relative) timestamps as features, so each pose and radar frame should have the associated timestamp (if we are exporting a specialised re-synchronisation mapping then we can use predicted timestamps). Also, we should saving num_people alongside each Kinect/pose frame/sample. 

In the future we might implement some kind of people tracking feature for frames where there are multiple people. For now, we are only going to be training single person pose estimation models. So we can export data with frames involving single and multiple people and keep the current per-frame pose ordering, but during training we will likely filter-out cases where there are multiple people in the frame. 

Please also generate a short jupyter notebook explaining the meaning of the exported data, how to use the exported data, and what each exported data field means. 


## Deployed SyncWB
* Do they only need to run normal launch? When do they run the full asset scan? Does the deployed SyncWB automatically perform the asset scan on the first time it is launched?
  * Full package checksums and SQLite integrity checks performed during setup
  * Lightweight package/assignment identity checks are performed during normal GUI launch (nothing special happens on first launch except for asking for annotator ID)

Must disclose during handover:
* Annotator id
General:
* Make it more concise. 



Must not omit this information from the SyncWB guides:

* Elaborate slightly more on how the initial temporal mapping derived (1 more sentence), explaining how this can be inaccurate and subject to drift.
* Describe how I will evaluate anchors using the leave one anchor out method, but word this as a way to motivate students to be precise and accurate when placing anchors and not be careless. 
* While there isn't a clear answer to 'how many anchors should I place', students should try to aim for 20-30 clear anchors including near the start and end, though we prefer reliable anchors over additional uncertain ones. 
* Explain synchronisation algorithm to an extent such that students understand the importance of labelling start/end of recordings. 
* Describe how we have spatial calibration between Kinect and radar, but it might not be perfect and we may improve this further in the near future, so if there is a visible difference between location of 3D pose keypoints and location of radar cloud points, this can't always be explained as temporal synchronisation error. 
* Recommend Doppler based colouring to infer motion segments, points will be coloured red if the object/person is moving away from the radar, blue if moving towards the radar, green if stationary. 






