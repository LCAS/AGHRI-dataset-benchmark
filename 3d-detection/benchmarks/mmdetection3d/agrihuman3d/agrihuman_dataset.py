import numpy as np

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class AgriHumanLidarDataset(Det3DDataset):
    METAINFO = {
        'classes': ('person', ),
        'palette': [(220, 20, 60)],
    }

    def parse_ann_info(self, info):
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        ann_info = self._remove_dontcare(ann_info)
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0.0),
        )
        return ann_info
