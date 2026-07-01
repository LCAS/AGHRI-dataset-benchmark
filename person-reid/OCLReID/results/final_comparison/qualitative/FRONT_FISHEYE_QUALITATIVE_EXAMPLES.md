# Front Fisheye Qualitative Examples

These qualitative examples preserve the original selected frames and saved inference overlays. Only the displayed part-OCLReID label was changed to Normal part-OCLReID.

Contact sheet: `/home/prabuddhi/Desktop/OCLReID/results/final_test_core_metrics_summary_with_gate/figures/fig8_front_fisheye_qualitative_examples.png`

## Footpath

- Scene: `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label`
- Camera: `cam_fish_front`
- Target class: 01
- Selected video frame: 166
- Why selected: front-fisheye footpath case where Normal part-OCLReID localizes the distant target while rpf-ReID is lost
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/cam_fish_front/class_01/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/cam_fish_front/class_01/inference_visualization.mp4`

## Polytunnel

- Scene: `in_straw_3pick_diff_st_10_24_2024_5_a_label`
- Camera: `cam_fish_front`
- Target class: 02
- Selected video frame: 37
- Why selected: front-fisheye neighbouring-row target with stronger Normal part-OCLReID confidence than rpf-ReID
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label/cam_fish_front/class_02/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label/cam_fish_front/class_02/inference_visualization.mp4`

## Vineyard

- Scene: `out_vine_4swap+walk_st_ly_11_06_2024_2_label`
- Camera: `cam_fish_front`
- Target class: 02
- Selected video frame: 623
- Why selected: front-fisheye vineyard case where Normal part-OCLReID localizes an occluded target while rpf-ReID is lost
- rpf-ReID MP4: `archived rpf-ReID full result root/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_fish_front/class_02/inference_visualization.mp4`
- Normal part-OCLReID MP4: `archived Normal part-OCLReID full result root/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_fish_front/class_02/inference_visualization.mp4`
