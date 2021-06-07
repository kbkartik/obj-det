# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup

import detectron2
import xlsxwriter as xw
import matplotlib.pyplot as plt

# import some common libraries
import numpy as np
import cv2
import random
import json
from detectron2.structures import BoxMode
#from google.colab.patches import cv2_imshow

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    #for dataset_name in cfg.DATASETS.TEST:
    dataset_name = (cfg.DATASETS.TEST)[0]
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    )
    results_i = inference_on_dataset(model, data_loader, evaluator)
    #results_i, valid_losses_reduced, valid_loss_dict_reduced = inference_on_dataset(model, data_loader, evaluator) # Validation loss
        
    #results[dataset_name] = results_i
    if comm.is_main_process():
       temp_dict = results_i['bboxcmdict']
       del results_i['bboxcmdict']
       logger.info("Evaluation results for {} in csv format:".format(dataset_name))
       print_csv_format(results_i)
       results_i['bboxcmdict'] = temp_dict
    if len(results_i) == 1:
        results_i = list(results_i.values())[0]
    return results_i #, valid_losses_reduced, valid_loss_dict_reduced

def do_train(cfg, model, resume=False):

    default_val_AP = 0
    default_val_AP50 = 0
    default_val_AP75 = 0
    best_model_dict = {}
    val_stat_cm_dict = {}
    
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    terminal_writer = (
        [
            CommonMetricPrinter(max_iter),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                #and iteration != max_iter
            ):
                val_dict = do_test(cfg, model)
                #val_dict, valid_losses_reduced, valid_loss_dict_reduced = do_test(cfg, model) # Validation losses

                if comm.is_main_process():
                
                   if (val_dict['bbox']['AP'] > default_val_AP and val_dict['bbox']['AP50'] > default_val_AP50 and val_dict['bbox']['AP75'] > default_val_AP75):

                      default_val_AP = val_dict['bbox']['AP']
                      default_val_AP50 = val_dict['bbox']['AP50']
                      default_val_AP75 = val_dict['bbox']['AP75']
                      
                      if not bool(val_dict['bboxcmdict']):
                         print("Empty state dict ", iteration)
                      
                      else:
                         val_stat_cm_dict = val_dict['bboxcmdict']

                      best_model_dict = {}

                      best_model_dict["model"] = model.state_dict()
                      best_model_dict["optimizer"] = optimizer.state_dict()
                      best_model_dict["scheduler"] = scheduler.state_dict()
                    
                      #print("MAP: {} iteration: {} lr: {} length: {}".format(default_validation_map, iteration,
                      #       best_model_dict["optimizer"]["param_groups"][0]['lr'], len(best_model_dict["optimizer"]["param_groups"])))
                      torch.save(best_model_dict, cfg.OUTPUT_DIR+'model_dict_'+str(iteration)+'.pth') 
                
                # Validation losses
                #if comm.is_main_process():
                #    storage.put_scalars(valid_total_loss=valid_losses_reduced, **valid_loss_dict_reduced)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 2631 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            if iteration - start_iter > 5 and (iteration % 300 == 0 or iteration == max_iter):
                for wrt in terminal_writer:
                    wrt.write()
            if comm.is_main_process():
                periodic_checkpointer.step(iteration)

    if bool(val_stat_cm_dict):
       dict_to_xlsx(val_stat_cm_dict)

def dict_to_xlsx(cmdict):

    wb = xw.Workbook('/ssd_scratch/cvit/myfolder/idd_data_coco/models/bestcmstat.xlsx')
    wk = wb.add_worksheet()
    row = 0
    col = 0

    wk.write(0, 1, "Total Pred")
    wk.write(0, 2, "True Pos")
    wk.write(0, 3, "False Pos")
    wk.write(0, 4, "False Neg")
    wk.write(0, 5, "Total GT")

    order = sorted(cmdict.keys())
    for key in order:
       row += 1
       wk.write(row, col, key)
       i = 1
       for item in cmdict[key]:
           wk.write(row, col + i, item)
           i += 1

       if (row+1) % 3 == 0:
          row += 1
          wk.write(row, col, None)

    wb.close()
    
def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = '/ssd_scratch/cvit/myfolder/idd_data_coco/models/'
    cfg.MODEL.MASK_ON = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 12 # 16
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 832, 864, 896, 928, 960, 992, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.DATASETS.TRAIN = ("idd_train",)
    cfg.DATASETS.TEST = ("idd_val",)
    cfg.SOLVER.MAX_ITER = 105240 # 98650
    cfg.SOLVER.CHECKPOINT_PERIOD = 2631 # 1973
    cfg.SOLVER.BASE_LR = 0.004 # pick a good LR 0.001, 0.005
    cfg.SOLVER.GAMMA = 0.25
    cfg.SOLVER.STEPS = (65775, 92085, ) # 78920
    cfg.TEST.EVAL_PERIOD = 10524 # 9865
    cfg.MODEL.WEIGHTS = "/ssd_scratch/cvit/myfolder/idd_data_coco/model_final_480dd8.pkl"
    #model_final_weights.pth" #CRCNN_model_surgery_coco_to_idd.pth" # Cascade model_final_480dd8.pkl
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # default: 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.BOUNDARY_THRESH = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    #cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 2

    """
    #cfg.SOLVER.MOMENTUM = 0.9
    #cfg.SOLVER.LR_SCHEDULER_NAME = "CyclicLR"
    #cfg.SOLVER.MIN_LR = 0.0008
    #cfg.SOLVER.MAX_LR = 0.0025
    #cfg.SOLVER.STEP_SIZE_UP = 11838
    #cfg.SOLVER.MAX_MOMENTUM = 0.9
    """

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code

    register_coco_instances("idd_train", {}, "/ssd_scratch/cvit/myfolder/idd_data_coco/idd_train_annotation.json", "/ssd_scratch/cvit/myfolder/idd_data_coco/idd_train_imgs")
    register_coco_instances("idd_val", {}, "/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation.json", "/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_imgs")
    return cfg

def remove_solver_states(model_path):

    model = torch.load(model_path)
    del model["optimizer"]
    del model["scheduler"]
    if "iteration" in model.keys():
       del model["iteration"]
    print(list(model.keys()))
    torch.save(model, '/ssd_scratch/cvit/myfolder/idd_data_coco/models_run1/model_only_weights.pth')
    print("Remove solver states complete")

def main(args):

    #Removing solver states from model_final which contains every checkpointable object
    #model_path = '/ssd_scratch/cvit/myfolder/idd_data_coco/models_run1/model_final.pth'
    #if comm.is_main_process():
    #   remove_solver_states(model_path)

    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model)
    model_path = '/ssd_scratch/cvit/myfolder/idd_data_coco/models/model_final.pth'
    if comm.is_main_process():
        #remove_solver_states(model_path)
        print("Training Done")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
