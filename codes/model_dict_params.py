
import torch
import pickle

checkpoint_dict = torch.load("/home/kartik/Desktop/trial1.pth", map_location=torch.device('cpu'))
#checkpoint = torch.load("/home/kartik/Downloads/model_final_480dd8.pkl", map_location=torch.device('cpu'))
print("Checkpoint dict: ", list(checkpoint_dict.keys()))
"""
print("Optimizer dict: ", list(checkpoint_dict["optimizer"]))
#print("Optimizer state dict: ", list(checkpoint_dict["optimizer"]["state"].keys()))
first_optimizer_state = checkpoint_dict["optimizer"]["state"][22833147181408]
print("momentum buffer of first_optimizer_state: ", first_optimizer_state)
print("Optimizer param_groups type and list: ", type(checkpoint_dict["optimizer"]["param_groups"]),
      len(checkpoint_dict["optimizer"]["param_groups"]))
first_optimizer_param_groups = checkpoint_dict["optimizer"]["param_groups"][87]
print("Internal optimizer param_groups", first_optimizer_param_groups)
"""
m = checkpoint_dict['model']

weight_names = {
        "module.roi_heads.box_predictor.0.cls_score.weight",
        "module.roi_heads.box_predictor.1.cls_score.weight",
        "module.roi_heads.box_predictor.2.cls_score.weight",
        "module.roi_heads.mask_head.predictor.weight",
}
bias_names = {
        "module.roi_heads.box_predictor.0.cls_score.bias",
        "module.roi_heads.box_predictor.1.cls_score.bias",
        "module.roi_heads.box_predictor.2.cls_score.bias",
        "module.roi_heads.mask_head.predictor.bias",
}

for key, val in m.items():
    if key in weight_names or key in bias_names:
        print(key, m[key].shape)
