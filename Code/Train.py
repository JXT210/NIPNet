import os
import random
import numpy as np
import torch
import torch.utils.data as Data
from Functions import generate_grid, Dataset_epoch, Predict_dataset, SpatialTransform, SpatialTransformNearest
from Loss import smoothloss, NCC, compute_label_dice, PDLoss
from Model import NIPNet
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=100000,
                    help="number of total iteration")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--pyramid_distillation_weight", type=float, help="pyramid_distillation_weight",
                    dest="pyramid_distillation_weight", default=0.1)
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="")
parser.add_argument("--val_dir", type=str, help="data folder with val vols",
                    dest="val_dir", default="")
parser.add_argument("--label_dir", type=str, help="data folder with val label vols",
                    dest="label_dir", default="")
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='../Log')
parser.add_argument("--model_dir", type=str, help="model folder",
                    dest="model_dir", default='../Model')
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=2000,
                    help="frequency of saving models")

opt = parser.parse_args()
lr = opt.lr
iteration = opt.iteration
train_dir = opt.train_dir
val_dir = opt.val_dir
label_dir = opt.label_dir
log_dir = opt.log_dir
model_dir = opt.model_dir
smooth = opt.smooth
pyramid_distillation_weight = opt.pyramid_distillation_weight

log_txt = "../Log/NIPNet_update.txt"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
with open(log_txt, "w") as log:
    log.write("Validation Dice log for NIPNet:\n")


def train():
    device = torch.device('cuda')
    model = NIPNet(dim=3, device=device, imgshape=imgshape).to(device)
    loss_similarity = NCC(win=9)
    loss_smooth = smoothloss
    loss_pdl = PDLoss
    # 日志文件
    log_name = "NIPNet-iteration-" + str(iteration) + "-lr-" + str(lr)
    print("log_name: ", log_name)
    f = open(os.path.join(log_dir, log_name + ".txt"), "w")

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    transform = SpatialTransform().to(device)
    transform_nearest = SpatialTransformNearest().to(device)
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    for param in transform_nearest.parameters():
        param.requires_grad = False
        param.volatile = True

    training_loader = Data.DataLoader(Dataset_epoch(train_dir), batch_size=1, shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(Predict_dataset(val_dir, label_dir), batch_size=1, shuffle=False, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    step = 0
    while step <= iteration:
        for X, Y in training_loader:
            # X fixed  Y moving
            X = X.to(device).float()
            Y = Y.to(device).float()
            flow, intermediate_flows = model(X, Y)
            loss_main = loss_similarity(X, transform(Y, flow.permute(0, 2, 3, 4, 1), grid))
            smoothLoss = loss_smooth(flow)
            pdlLoss = loss_pdl(flow, intermediate_flows)
            loss = loss_main + smoothLoss * smooth + pyramid_distillation_weight * pdlLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step % opt.checkpoint == 0):
                model_name = model_dir + '/NIPNet_' + str(step) + '.pth'
                torch.save(model.state_dict(), model_name)
                # Validation
                with torch.no_grad():
                    dice_total = []
                    for (X, Y, X_label, Y_label) in val_loader:
                        X = X.to(device)
                        Y = Y.to(device)
                        X_label = X_label.to(device)
                        Y_label = Y_label.to(device)
                        # X fixed   Y moving
                        flow, intermediate_flows = model(X, Y)

                        Y_X_label = transform_nearest(Y_label, flow.permute(0, 2, 3, 4, 1),
                                                      grid).data.cpu().numpy()[0, 0, :, :, :]
                        X_label = X_label.data.cpu().numpy()[0, 0, :, :, :]
                        dice_score = compute_label_dice(Y_X_label, X_label)
                        dice_total.append(dice_score)
                    dice_total = np.array(dice_total)
                    with open(log_txt, "a") as log:
                        log.write("Validing:" + str(step) + ":" + str(dice_total.mean()) + "\n")

            step += 1
            if step > iteration:
                break


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    same_seeds(123)
    imgshape = (160, 192, 224)
    train()
