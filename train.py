import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import datetime

from torch.nn.functional import mse_loss

from utils.dataset import get_data_loader_folder, get_paired_data_loader
from utils.utils import prepare_sub_folder, write_2images, write_html, print_params, adjust_learning_rate
from utils.MattingLaplacian import laplacian_loss_grad
from utils.losses import MSELoss
from utils.psnr_ssim import calculate_psnr,calculate_ssim, calculate_tuple
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument('--base_name', default=None, help='Directory name to save')
parser.add_argument('--mode', type=str, default='llie')
#parser.add_argument('--vgg_ckpoint', type=str, default='checkpoints/vgg_normalised.pth')

# Dataset
parser.add_argument('--train_content', default='../LLIEData/LOLv1/Train/input', help='Directory to dataset A')
parser.add_argument('--train_style', default='../LLIEData/LOLv1/Train/target', help='Directory to dataset B')
#parser.add_argument('--train_target', default='../../LOLv1/Train/target', help='Directory to dataset B')

parser.add_argument('--eval_content', default='../LLIEData/LOLv1/Test/input', help='Directory to dataset A')
parser.add_argument('--eval_style', default='../LLIEData/LOLv1/Test/target', help='Directory to dataset B')
#parser.add_argument('--eval_target', default='../../LOLv1/Test/target', help='Directory to dataset B')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--new_size', type=int, default=512)
parser.add_argument('--crop_size', type=int, default=256)

#parser.add_argument('--use_lap', type=bool, default=True)
#parser.add_argument('--win_rad', type=int, default=1, help='The larger the value, the more detail in the generated image and the higher the CPU and memory requirements (proportional to the win_rad**2)')

# Training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)

#parser.add_argument('--style_weight', type=float, default=1)
#parser.add_argument('--content_weight', type=float, default=0)
#parser.add_argument('--lap_weight', type=float, default=1500)
parser.add_argument('--rec_weight', type=float, default=0)
#parser.add_argument('--temporal_weight', type=float, default=60)

parser.add_argument('--training_iterations', type=int, default=16000) #160000
parser.add_argument('--fine_tuning_iterations', type=int, default=10000)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument('--resume_iter', type=int, default=-1)

# Log
parser.add_argument('--logs_directory', default='logs', help='Directory to log')
parser.add_argument('--display_size', type=int, default=15)
parser.add_argument('--eval_iter', type=int, default=1000)
#parser.add_argument('--image_save_iter', type=int, default=10000) #10000
parser.add_argument('--model_save_interval', type=int, default=4000) #10000
#parser.add_argument('--eval_iter', type=int, default=1000)

if __name__ =="__main__":
    args = parser.parse_args()
    if args.base_name is None:
        args.base_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # total_iterations = args.training_iterations + args.fine_tuning_iterations
    total_iterations = args.training_iterations
    current_iter = -1

    # Logs directory
    logs_directory = os.path.join(args.logs_directory, args.base_name)
    print("Logs directory:", logs_directory)
    checkpoint_directory, image_directory = prepare_sub_folder(logs_directory)

    # Dataset
    batch_size = args.batch_size
    num_workers = args.batch_size
    new_size = args.new_size
    crop_size = args.crop_size
    #win_rad = args.win_rad
    '''    train_loader_a = get_data_loader_folder(args.train_content, batch_size, new_size, crop_size, crop_size,
                                            use_lap=False, seed=42, num_workers=1)
    train_loader_c = get_data_loader_folder(args.train_style, batch_size, new_size, crop_size, crop_size, use_lap=False,
                                            seed=42, num_workers=1)
    train_loader_b = get_data_loader_folder(args.train_style, batch_size, new_size, crop_size, crop_size, use_lap=False,
                                            seed=1, num_workers=1, shuffle=True)

    test_loader_a = get_data_loader_folder(args.eval_content, 1, new_size, crop_size, crop_size, use_lap=False,
                                           seed=42, num_workers=1)
    test_loader_c = get_data_loader_folder(args.eval_style, 1, new_size, crop_size, crop_size, use_lap=False,
                                           seed=42, num_workers=1)
    test_loader_b = get_data_loader_folder(args.eval_style, 1, new_size, crop_size, crop_size, use_lap=False,
                                           seed=1, num_workers=1)'''
    train_loader = get_paired_data_loader(args.train_content, args.train_style,batch_size, new_size, crop_size, crop_size,
                                         num_workers=1)
    test_loader  =get_paired_data_loader(args.eval_content, args.eval_style,batch_size, None, crop_size, crop_size)

    print(len(test_loader))
    

    '''    # Reversible Network
    from models.RevResNet import RevResNet

    if args.mode.lower() == "llie":
        RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4,
                               hidden_dim=16, sp_steps=2)
    elif args.mode.lower() == "artistic":
        RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4,
                               hidden_dim=64, sp_steps=1)
    else:
        raise NotImplementedError()

    RevNetwork = RevNetwork.to(device)
    RevNetwork.train()
    print_params(RevNetwork)

    # Optimizer
    optimizer = torch.optim.Adam(RevNetwork.parameters(), lr=args.lr)

    # Transfer module
    from models.cWCT import cWCT

    cwct = cWCT(train_mode=True)

    

    # Resume 使用预训练参数对LLIE任务微调
    if args.resume:
        #state_dict = torch.load(os.path.join(checkpoint_directory, "last.pt"))
        state_dict = torch.load('./checkpoints/photo_image.pt')
        RevNetwork.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        current_iter = args.resume_iter
        print('Resume from %s. Resume iter is %d' % ('./checkpoints/photo_image.pt', args.resume_iter))

    # Loss
    

    # Training
    iter_loader_a, iter_loader_b, iter_loader_c = iter(train_loader_a), iter(train_loader_b), iter(train_loader_c)

    mseloss = MSELoss()


    while current_iter < total_iterations:
        images_a, images_b, images_c = next(iter_loader_a), next(iter_loader_b), next(iter_loader_c)

       

        images_a, images_b, images_c = images_a['img'].to(device), images_b['img'].to(device), images_c['img'].to(
            device)

        # Optimizer
        adjust_learning_rate(optimizer, args.lr, args.lr_decay, current_iter)
        optimizer.zero_grad()

        # Forward inference
        z_c = RevNetwork(images_a, forward=True)
        z_s = RevNetwork(images_b, forward=True)

        # Transfer
        try:
            z_cs = cwct.transfer(z_c, z_s)
        except:
            print('Cholesky Decomposition fails. Gradient infinity. Skip current batch.')
            with open(logs_directory + "/loss.log", "a") as log_file:
                log_file.write('Cholesky Decomposition fails. Gradient infinity. Skip current batch. \n')
            continue

        # Backward inference
        stylized = RevNetwork(z_cs, forward=False)

        # Style loss
        #loss_c, loss_s = vgg_enc(images_a, images_b, stylized, n_layer=4, content_weight=args.content_weight)

        # Cycle reconstruction
        if args.rec_weight > 0:
            z_cs = RevNetwork(stylized, forward=True)

            try:
                z_csc = cwct.transfer(z_cs, z_c)
            except:
                print('Cholesky Decomposition fails. Gradient infinity. Skip current batch.')
                with open(logs_directory + "/loss.log", "a") as log_file:
                    log_file.write('Cholesky Decomposition fails. Gradient infinity. Skip current batch. \n')
                continue

            rec = RevNetwork(z_csc, forward=False)
            loss_rec = mse_loss(rec, images_a)
        else:
            loss_rec = 0

        # Matting Laplacian loss
        

        # Temporal loss
        

        # Total loss
        # loss = args.content_weight * loss_c + args.style_weight * loss_s + args.rec_weight * loss_rec + args.temporal_weight * loss_tmp
        enhance_loss = mse_loss(stylized, images_a)
        loss = enhance_loss + args.rec_weight * loss_rec
        loss.backward()
        nn.utils.clip_grad_norm_(RevNetwork.parameters(), 5)
        optimizer.step()

        # Dump training stats in log file
        if (current_iter + 1) % 10 == 0:
            message = "Iteration: %08d/%08d  loss:%.4f enhance_loss:%.4f rec_loss:%.4f" % (
                # "Iteration: %08d/%08d  content_loss:%.4f  lap_loss:%.4f  rec_loss:%.4f  style_loss:%.4f  loss_tmp:%.4f  loss_tmp_GT:%.4f" % (
                current_iter + 1, total_iterations,
                loss,
                enhance_loss,
                loss_rec
                # args.content_weight * loss_c,
                # args.lap_weight * loss_lap,
                # args.rec_weight * loss_rec,
                # args.style_weight * loss_s,
                # args.temporal_weight * loss_tmp,
                # args.temporal_weight * loss_tmp_GT,
            )
            print(message)
            with open(logs_directory + "/loss.log", "a") as log_file:
                log_file.write('%s\n' % message)

            # Log sample
            if (current_iter + 1) % args.eval_iter == 0:
                cwct.train_mode = False
                with torch.no_grad():
                    # index = torch.randint(low=0, high=len(test_loader_a.dataset), size=[args.display_size])
                    index = len(test_loader_a.dataset)
                    test_display_images_a = torch.stack([test_loader_a.dataset[i]['img'] for i in range(index)])
                    # index = torch.randint(low=0, high=len(test_loader_b.dataset), size=[args.display_size])
                    test_display_images_b = torch.stack([test_loader_b.dataset[i-1]['img'] for i in range(index)])
                    test_display_images_c = torch.stack([test_loader_b.dataset[i]['img'] for i in range(index)])
                    test_image_outputs = RevNetwork.sample(cwct, test_display_images_a, test_display_images_b, device)
                    #print(type(test_image_outputs))
                eval_psnr = calculate_tuple(test_image_outputs[2], test_display_images_c, 'psnr')
                eval_ssim = calculate_tuple(test_image_outputs[2], test_display_images_c, 'ssim')
                #eval_psnr = calculate_psnr(test_image_outputs, test_display_images_c)
                #eval_ssim = calculate_ssim(test_image_outputs, test_display_images_c)
                print(f'Iteration {current_iter + 1} eval loss:psnr {eval_psnr},ssim {eval_ssim}')
                cwct.train_mode = True
                write_2images(test_image_outputs, args.display_size, image_directory, 'train_%08d' % (current_iter + 1))
                # HTML
                write_html(logs_directory + "/index.html", current_iter + 1, args.eval_iter, 'images')

            if (current_iter + 1) % args.image_display_iter == 0:
                cwct.train_mode = False
                with torch.no_grad():
                    index = torch.randint(low=0, high=len(train_loader_a.dataset), size=[args.display_size])
                    train_display_images_a = torch.stack([train_loader_a.dataset[i]['img'] for i in index])
                    index = torch.randint(low=0, high=len(train_loader_b.dataset), size=[args.display_size])
                    train_display_images_b = torch.stack([train_loader_b.dataset[i]['img'] for i in index])
                    image_outputs = RevNetwork.sample(cwct, train_display_images_a, train_display_images_b, device)
                cwct.train_mode = True
                write_2images(image_outputs, args.display_size, image_directory, 'train_current')

            # Save network weights
            if (current_iter + 1) % args.model_save_interval == 0:
                ckpoint_file = os.path.join(checkpoint_directory, f'{current_iter + 1}.pt')
                torch.save({'state_dict': RevNetwork.state_dict(), 'optimizer': optimizer.state_dict()}, ckpoint_file)

            if (current_iter + 1) == args.training_iterations:
                ckpoint_file = os.path.join(checkpoint_directory, 'model_LLIE.pt')
                torch.save({'state_dict': RevNetwork.state_dict()}, ckpoint_file)
            
        current_iter += 1

    print("Finishing training. Model save at %s" % checkpoint_directory)'''
