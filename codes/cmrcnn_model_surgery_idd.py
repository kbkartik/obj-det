#https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/configs/cityscapes

import torch
import torch.nn as nn

def clip_weights_from_pretrain_of_coco_to_idd(f, out_file):
    
    # COCO categories for pretty print
    COCO_CATEGORIES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # IDD of fine categories for pretty print
    IDD_FINE_CATEGORIES = [
        "__background__",
        "car",
        "bus",
        "autorickshaw",
        "vehicle fallback",
        "truck",
        "motorcycle",
        "rider",
        "person",
        "bicycle",
        "animal",
        "traffic sign",
        "train",
        "trailer",
        "traffic light",
        "caravan",
    ]
    
    coco_cats = COCO_CATEGORIES
    idd_cats = IDD_FINE_CATEGORIES
    
    coco_cats_to_inds = dict(zip(coco_cats, range(len(coco_cats))))
    idd_cats_to_inds = dict(
        zip(idd_cats, range(len(idd_cats)))
    )

    checkpoint = torch.load(f, map_location=torch.device('cpu'))
    m = checkpoint['model']

    weight_names = {
        "roi_heads.cls_score.0": "module.roi_heads.box_predictor.0.cls_score.weight",
        "roi_heads.cls_score.1": "module.roi_heads.box_predictor.1.cls_score.weight",
        "roi_heads.cls_score.2": "module.roi_heads.box_predictor.2.cls_score.weight",
    }
    bias_names = {
        "roi_heads.cls_score.0": "module.roi_heads.box_predictor.0.cls_score.bias",
        "roi_heads.cls_score.1": "module.roi_heads.box_predictor.1.cls_score.bias",
        "roi_heads.cls_score.2": "module.roi_heads.box_predictor.2.cls_score.bias",
    }
    
    representation_size_cls_score_0 = m[weight_names["roi_heads.cls_score.0"]].size(1)
    representation_size_cls_score_1 = representation_size_cls_score_0
    representation_size_cls_score_2 = representation_size_cls_score_0
        
    cls_score_0 = nn.Linear(representation_size_cls_score_0, len(idd_cats))
    cls_score_1 = nn.Linear(representation_size_cls_score_1, len(idd_cats))
    cls_score_2 = nn.Linear(representation_size_cls_score_2, len(idd_cats))
    
    nn.init.normal_(cls_score_0.weight, std=0.01)
    nn.init.constant_(cls_score_0.bias, 0)
    nn.init.normal_(cls_score_1.weight, std=0.01)
    nn.init.constant_(cls_score_1.bias, 0)
    nn.init.normal_(cls_score_2.weight, std=0.01)
    nn.init.constant_(cls_score_2.bias, 0)
    
    def _copy_weight(src_weight, dst_weight, cityscp_cats):        
        for ix, cat in enumerate(cityscp_cats):
            if cat not in coco_cats:
                continue
            jx = coco_cats_to_inds[cat]
            dst_weight[ix] = src_weight[jx]
        return dst_weight

    def _copy_bias(src_bias, dst_bias, cityscp_cats, class_agnostic=False):
        if class_agnostic:
            return dst_bias
        return _copy_weight(src_bias, dst_bias, cityscp_cats)

    print(m[weight_names["roi_heads.cls_score.0"]].shape)
    m[weight_names["roi_heads.cls_score.0"]] = _copy_weight(
        m[weight_names["roi_heads.cls_score.0"]], cls_score_0.weight, idd_cats
    )
    print(m[weight_names["roi_heads.cls_score.0"]].shape)
    
    print(m[weight_names["roi_heads.cls_score.1"]].shape)
    m[weight_names["roi_heads.cls_score.1"]] = _copy_weight(
        m[weight_names["roi_heads.cls_score.1"]], cls_score_1.weight, idd_cats
    )
    print(m[weight_names["roi_heads.cls_score.1"]].shape)

    print(m[weight_names["roi_heads.cls_score.2"]].shape)
    m[weight_names["roi_heads.cls_score.2"]] = _copy_weight(
        m[weight_names["roi_heads.cls_score.2"]], cls_score_2.weight, idd_cats
    )
    print(m[weight_names["roi_heads.cls_score.2"]].shape)

    print(m[bias_names["roi_heads.cls_score.0"]].shape)
    m[bias_names["roi_heads.cls_score.0"]] = _copy_bias(
        m[bias_names["roi_heads.cls_score.0"]], cls_score_0.bias, idd_cats
    )
    print(m[bias_names["roi_heads.cls_score.0"]].shape)

    print(m[bias_names["roi_heads.cls_score.1"]].shape)
    m[bias_names["roi_heads.cls_score.1"]] = _copy_bias(
        m[bias_names["roi_heads.cls_score.1"]], cls_score_1.bias, idd_cats
    )
    print(m[bias_names["roi_heads.cls_score.1"]].shape)

    print(m[bias_names["roi_heads.cls_score.2"]].shape)
    m[bias_names["roi_heads.cls_score.2"]] = _copy_bias(
        m[bias_names["roi_heads.cls_score.2"]], cls_score_2.bias, idd_cats
    )
    print(m[bias_names["roi_heads.cls_score.2"]].shape)

    print("f: {}\nout_file: {}".format(f, out_file))

    new_checkpoint = {}
    new_checkpoint['model'] = m
    
    torch.save(new_checkpoint, out_file)
    
    print("Done")

clip_weights_from_pretrain_of_coco_to_idd("/home/kartik/Downloads/Model_CMRCNN_pretrained_COCO.pth", "/home/kartik/Desktop/trial1.pth")
