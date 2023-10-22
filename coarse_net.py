import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchsummary import summary
from torch.nn import functional as F
from utils.loss import GANLoss, AdversarialLoss, PerceptualLoss, sift_loss_l1

gpu_id = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Residual(BaseNetwork):
    def __init__(self, input_channels, mid_channels, out_channels, stride=1, skip=True):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, mid_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        if skip:
            self.layer5 = nn.Conv2d(
                input_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.layer5 = None
        self.init_weights()

    def forward(self, data):
        layer1 = self.layer1(data)
        layer2 = self.layer2(layer1)
        layer4 = self.layer4(layer2)
        if self.layer5:
            ski = self.layer5(data)
            out = F.leaky_relu((layer4 + ski), negative_slope=0.2)
        else:
            out = F.leaky_relu(layer4, negative_slope=0.2)
        return out


class Block(BaseNetwork):
    def __init__(self, input_channels=128, mid_channels=128, out_channels=256):
        super().__init__()
        self.block1 = Residual(input_channels, mid_channels, mid_channels)
        self.block2 = Residual(mid_channels, out_channels, out_channels)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, data):
        block1 = self.block1(data)
        block2 = self.block2(block1)
        out = self.pooling(block2)
        return out, block2


class De_residual(BaseNetwork):
    def __init__(self, input_channels, mid_channels, out_channels, stride=1, skip=True):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, mid_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        if skip:
            self.layer5 = nn.Conv2d(
                input_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.layer5 = None
        self.init_weights()

    def forward(self, data):
        layer1 = self.layer1(data)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        if self.layer5:
            ski = self.layer5(data)
            out = F.leaky_relu((layer4 + ski), negative_slope=0.2)
        else:
            out = F.leaky_relu(layer4, negative_slope=0.2)
        return out


class De_block(BaseNetwork):
    def __init__(self, input_channels, mid_channels, out_channels, use_connect=True):
        super().__init__()
        self.block1 = De_residual(input_channels, mid_channels, mid_channels)
        self.block2 = De_residual(input_channels, out_channels, out_channels)
        self.unpooling = nn.UpsamplingNearest2d(scale_factor=2)
        self.use_connect = use_connect

    def forward(self, data, connect=None):
        temp = self.unpooling(data)
        if self.use_connect:
            block1 = self.block1(torch.cat([temp, connect], dim=1))
            block2 = self.block2(torch.cat([block1, connect], dim=1))
        else:
            block1 = self.block1(temp)
            block2 = self.block2(block1)
        return block2


class Img_generator(BaseNetwork):
    def __init__(self):
        super(Img_generator, self).__init__()
        self.block1 = Block(128, 128, 256)
        self.block2 = Block(256, 256, 512)
        self.block3 = Block(512, 512, 512)
        self.block4 = Block(512, 512, 512)
        self.block5 = Block(512, 512, 512)
        self.block6 = Block(512, 512, 512)
        self.deblock1 = De_block(1024, 512, 512)
        self.deblock2 = De_block(1024, 512, 512)
        self.deblock3 = De_block(1024, 512, 512)
        self.deblock4 = De_block(1024, 512, 256)
        self.deblock5 = De_block(768, 256, 128)
        self.deblock6 = De_block(384, 128, 64)
        self.deblock7 = De_block(64, 64, 3, use_connect=False)

    def forward(self, sift):
        block1, connect1 = self.block1(sift)
        block2, connect2 = self.block2(block1)
        block3, connect3 = self.block3(block2)
        block4, connect4 = self.block4(block3)
        block5, connect5 = self.block5(block4)
        block6, connect6 = self.block6(block5)
        deblock1 = self.deblock1(block6, connect6)
        deblock2 = self.deblock2(deblock1, connect5)
        deblock3 = self.deblock3(deblock2, connect4)
        deblock4 = self.deblock4(deblock3, connect3)
        deblock5 = self.deblock5(deblock4, connect2)
        deblock6 = self.deblock6(deblock5, connect1)
        deblock7 = self.deblock7(deblock6)
        return deblock7

# net=Img_generator().cuda()
# summary(net,(128,128,128))

class Discriminator(BaseNetwork):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.level1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.level7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, x):
        level1 = self.level1(x)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        level4 = self.level4(level3)
        level5 = self.level5(level4)
        level6 = self.level6(level5)
        level7 = self.level7(level6)
        return level7

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_dir = 'weights/'

class ImgModel(BaseModel):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.lr = 1e-4
        self.gen = torch.nn.DataParallel(Img_generator()).cuda()
        self.dis = torch.nn.DataParallel(Discriminator(3)).cuda()
        self.gan_type = 're_avg_gan'
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss()
        self.perceptual_loss = PerceptualLoss()
        self.criterionGAN = GANLoss(gan_type=self.gan_type)
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.ADV_LOSS_WEIGHT = 0.02
        self.L1_LOSS_WEIGHT = 1
        self.PERC_LOSS_WEIGHT = 0.01
        self.SIFT_LOSS_WEIGHT = 0.01

    def process(self, sift, truth):
        fake = self(sift)
        dis_fake = self.dis(fake.detach())
        dis_real = self.dis(truth)
        self.dis_optimizer.zero_grad()
        dis_loss = self.criterionGAN(dis_real - dis_fake, True)
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss = self.criterion(fake, truth)
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return fake, gen_loss

    def criterion(self, fake, truth):
        gen_fake = self.dis(fake)
        gen_real = self.dis(truth)
        gen_gan_loss = (self.criterionGAN(gen_real - torch.mean(gen_fake), False) +
                        self.criterionGAN(gen_fake - torch.mean(gen_real), True)) / 2. * self.ADV_LOSS_WEIGHT
        gen_perceptual_loss = self.perceptual_loss(fake, truth) * self.PERC_LOSS_WEIGHT
        gen_l1_loss = self.l1_loss(fake, truth) * self.L1_LOSS_WEIGHT
        gen_sift_loss = sift_loss_l1(truth, fake) * self.SIFT_LOSS_WEIGHT

        gen_loss = gen_gan_loss + gen_l1_loss + gen_perceptual_loss + gen_sift_loss
        return gen_loss

    def forward(self, sift):
        return self.gen(sift)

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'img_gen.pth')
        torch.save(self.dis.state_dict(), self.save_dir + path + 'img_dis.pth')

    def load(self, path):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'img_gen.pth'))
