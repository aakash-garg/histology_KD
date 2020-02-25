import torch
import src.vision as vision

model_dict = {
    "resnet8_v": vision.resnet8,  # params: 95082
    "resnet14_v": vision.resnet14,  # params: 192298
    "resnet20_v": vision.resnet20,  # params: 289514
    "resnet10_v": vision.resnet10,  # params: 4910922
    "resnet18_v": vision.resnet18,  # params: 11181642
    "resnet34_v": vision.resnet34,  # params: 21289802
    "resnet50_v": vision.resnet50,  # params: 23528522
    "resnet101_v": vision.resnet101,  # params: 42520650
    "resnet152_v": vision.resnet152,  # params: 58164298
    "wrn50_2": vision.wide_resnet50_2,  # params: 66854730
    "wrn101_2": vision.wide_resnet101_2,  # params: 124858186
}


def create_model(args):
    model_cls = model_dict[args.resnet]
    print(f"Building model {args.resnet}...", end='')
    model = model_cls(num_classes=args.num_classes, pretrained=args.pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    layers = len(list(model.modules()))
    print(f" total parameters: {total_params}, layers {layers}")
    # always use dataparallel for now
    model = torch.nn.DataParallel(model)
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPU(s).")
    # copy to cuda if activated
    if device_count > 0:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model


# if __name__ == "__main__":
#     for model in model_dict.keys():
#         create_model(args, "cpu")
