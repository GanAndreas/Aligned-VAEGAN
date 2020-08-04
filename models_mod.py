import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight,gain=0.5)

        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class encoder_template(nn.Module):

    def __init__(self,input_dim,latent_size,hidden_size_rule,device):
        super(encoder_template,self).__init__()



        if len(hidden_size_rule)==2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule)==3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1] , latent_size]

        modules = []
        for i in range(len(self.layer_sizes)-2):

            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.ReLU())

        self.feature_encoder = nn.Sequential(*modules)

        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)


        self.apply(weights_init)

        self.to(device)


    def forward(self,x):

        h = self.feature_encoder(x)


        mu =  self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar

class decoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,hidden_size_rule,device):
        super(decoder_template,self).__init__()


        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)

        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc, device):
        super(Discriminator, self).__init__()
        dis_model1 = [
            nn.Linear(input_nc, 4096, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 1024, bias=False),
            nn.LeakyReLU(0.2, True)]
        dis_model2 = [
            nn.Linear(1024, 1, bias=False),
            nn.LeakyReLU(0.2, True)]

        self.dis_model1 = nn.Sequential(*dis_model1)
        self.dis_model2 = nn.Sequential(*dis_model2)

        self.apply(weights_init)

        self.to(device)

    def forward(self, input):
        return self.dis_model2(self.dis_model1(input))

class Discriminator_special(nn.Module):
    def __init__(self, input_nc, device):
        super(Discriminator_special, self).__init__()
        dis_model1 = [
            nn.Linear(input_nc, 4096, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 1024, bias=False),
            nn.LeakyReLU(0.2, True)]
        dis_model2 = [
            nn.Linear(1024, 1, bias=False),
            nn.LeakyReLU(0.2, True)]

        self.dis_model1 = nn.Sequential(*dis_model1)
        self.dis_model2 = nn.Sequential(*dis_model2)

        self.apply(weights_init)

        self.to(device)

    def forward(self, input):
        return self.dis_model1(input)

class Discriminator_woConcat(nn.Module):
    def __init__(self, input_nc, device):
        super(Discriminator_woConcat, self).__init__()
        dis_model1 = [
            nn.Linear(input_nc, 64, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 128, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 256, bias=False),
            nn.LeakyReLU(0.2, True)]

        dis_model2 = [
            nn.Linear(256, 1, bias=False),
            nn.LeakyReLU(0.2, True)]

        self.dis_model1 = nn.Sequential(*dis_model1)
        self.dis_model2 = nn.Sequential(*dis_model2)

        self.apply(weights_init)

        self.to(device)

    def forward(self, input):
        return self.dis_model1(input)
