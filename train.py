from src import *
import torch.nn as nn

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.distill:
    student_network = create_model(args, student=True)
    teacher_network = create_model(args, student=False)
    model = DistillationModel(args, student_network, teacher_network)
    if args.attention_transfer:
        model.train_at()
    else:
        model.train_kd()

else:
    pw_network = create_model(args, student=True)
    pw_model = PatchWiseModel(args, pw_network)
    pw_model.train()
