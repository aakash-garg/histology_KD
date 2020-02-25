from src import *
import torch.nn as nn

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# pw_network = PatchNet(model=args.resnet, pretrained=True)
pw_network = create_model(args)
# pw_network = nn.DataParallel(pw_network, device_ids=[int(x) for x in args.gpu_ids.split(',')])
pw_model = PatchWiseModel(args, pw_network)
pw_model.train()
