# ZED Front Qualitative Examples

These qualitative examples preserve the original selected frames and saved inference overlays. Only the displayed part-OCLReID label was changed to Normal part-OCLReID.

Contact sheet: `/home/prabuddhi/Desktop/OCLReID/results/final_test_core_metrics_summary_with_gate/figures/fig5_zed_front_qualitative_examples.png`

## Footpath

- Scene: `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label`
- Camera: `cam_zed_rgb`
- Target class: 01
- Selected video frame: 174
- Why selected: clear visible target with Normal part-OCLReID correctly localized
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/cam_zed_rgb/class_01/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/cam_zed_rgb/class_01/inference_visualization.mp4`

## Polytunnel

- Scene: `in_straw_3pick_diff_st_10_24_2024_5_a_label`
- Camera: `cam_zed_rgb`
- Target class: 06
- Selected video frame: 149
- Why selected: target correctly localized with another detected person visible in the neighbouring row
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label/cam_zed_rgb/class_06/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label/cam_zed_rgb/class_06/inference_visualization.mp4`

## Vineyard

- Scene: `out_vine_4swap+walk_st_ly_11_06_2024_2_label`
- Camera: `cam_zed_rgb`
- Target class: 05
- Selected video frame: 35
- Why selected: Normal part-OCLReID correctly localizes a more occluded target while rpf-ReID has no target detection on the same frame
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_zed_rgb/class_05/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_zed_rgb/class_05/inference_visualization.mp4`
