import os
import cv2
import argparse
import numpy as np
import data
import coarse_net
from CFGAN import ImgModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from utils.FID import fid

class SIFTReconstruction():
    def __init__(self):
        self.train_num = 20
        self.test_num = 10
        self.batch_size = 4
        self.n_epochs = 100
        train_file = 'dataset/train/'
        test_file = 'dataset/test/'
        train_dataset = data.SIFTDataset(self.train_num, train_file)
        test_dataset = data.SIFTDataset(self.test_num, test_file)
        self.img_model = ImgModel().cuda()
        self.prior = coarse_net.ImgModel().cuda()
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    def train(self):
        print('\nTrain/Val ' + ' model:')
        for epoch in range(self.n_epochs):
            gen_losses = []
            print('epoch:' + str(epoch) + '********************************************')
            for cnt, items in enumerate(self.train_loader):
                self.img_model.train()
                self.prior.eval()
                truth, sift = (item.cuda() for item in items[:-1])
                prior=self.prior(sift)
                fake, gen_img_loss = self.img_model.process(sift, truth,prior)
                gen_losses.append(gen_img_loss.item())
            print('Tra (%d) train_loss:%5.4f' % (epoch, np.mean(gen_losses)))
            eval_loss = self.test()
            print('eval_loss: ' + str(eval_loss))
        self.img_model.save('CFGAN' + '/')

    def test(self, pretrained=False):
        if pretrained:
            self.img_model.eval()
            ssim, psnr = [], []
            print('\nTest ' + ' model:')
            self.img_model.load('CFGAN' + '/')
            self.prior.load('coarse_net'+'/')
            if not os.path.exists('res/CFGAN'):
                os.makedirs('res/CFGAN')
            for cnt, items in enumerate(self.test_loader):
                truth, sift = (item.cuda() for item in items[:-1])
                prior=self.prior(sift)
                fake = self.img_model(sift,prior)
                s, p = self.metrics(truth, fake)
                ssim.append(s)
                psnr.append(p)
                if cnt < 100:
                    fake = self.postprocess(fake)
                    path = 'res/CFGAN/' + 'Io_%06d.jpg' % (cnt + 1)
                    cv2.imwrite(path, fake[0])
            path1 = 'dataset/test/'
            path2 = 'res/CFGAN/'
            path = [path1, path2]
            fid_result = fid(path)
            print(' Evaluation: SSIM:%4.4f, PSNR:%4.2f, FID:%4.2f' % (np.mean(ssim), np.mean(psnr), fid_result))
        self.img_model.eval()
        val_loss = []
        for cnt, items in enumerate(self.test_loader):
            truth, sift = (item.cuda() for item in items[:-1])
            prior=self.prior(sift)
            fake = self.img_model(sift,prior)
            loss = self.img_model.criterion(fake, truth)
            val_loss.append(loss.item())
        return np.mean(val_loss)

    def postprocess(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int().cpu().detach().numpy()

    def metrics(self, Ig, Io):
        a = self.postprocess(Ig)
        b = self.postprocess(Io)
        ssim, psnr = [], []
        for i in range(len(a)):
            ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, channel_axis=2))
            psnr.append(compare_psnr(a[i], b[i], data_range=255))
        return np.mean(ssim), np.mean(psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    args = parser.parse_args()
    model = SIFTReconstruction()
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test(True)
    print('End.')

