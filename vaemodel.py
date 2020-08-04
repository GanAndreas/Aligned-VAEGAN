#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils import data
from tensorboardX import SummaryWriter
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models
import itertools
import utils

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()

        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader( self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= self.device )
        self.writer = SummaryWriter()
        self.num_gen_iter = hyperparameters['num_gen_iter']
        self.num_dis_iter = hyperparameters['num_dis_iter']
        self.pretrain = hyperparameters['pretrain']

        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources,feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)

            print(str(datatype) + ' ' + str(dim))
        
        print('latent size ' + str(self.latent_size))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())

        # The discriminator network is defined here
        self.net_D_Att = models.Discriminator(self.dataset.aux_data.size(1) + 2048 , self.device)
        self.net_D_Img = models.Discriminator(2048 + self.dataset.aux_data.size(1), self.device)

        self.optimizer_G = optim.Adam(parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=True)
        self.optimizer_D = optim.Adam(itertools.chain(self.net_D_Att.parameters(), self.net_D_Img.parameters()), lr=hyperparameters['lr_gen_model'], betas=(0.5, 0.999), weight_decay=0.0005)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(reduction='sum')

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(reduction='sum')

        self.MSE = nn.MSELoss(reduction='sum')
        self.L1 = nn.L1Loss(reduction='sum')

        self.att_fake_from_att_sample = utils.Sample_from_Pool()
        self.att_fake_from_img_sample = utils.Sample_from_Pool()
        self.img_fake_from_img_sample = utils.Sample_from_Pool()
        self.img_fake_from_att_sample = utils.Sample_from_Pool()

        if self.generalized:
            print('mode: gzsl')
            self.clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            self.clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)

    # reparameterize trick for sampling the latent distribution
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).to(self.device).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def trainstep(self, img, att, epoch, iters):

        for gen_iter in range(0, self.num_gen_iter):
            ##############################################
            # Training the VAE/Generator Network
            ##############################################

            ##############################################
            # Encode image features and attributes
            ##############################################

            utils.set_grad([self.net_D_Img, self.net_D_Att], False)
            self.optimizer_G.zero_grad()

            mu_img, logvar_img = self.encoder['resnet_features'](img)
            z_from_img = self.reparameterize(mu_img, logvar_img)

            mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
            z_from_att = self.reparameterize(mu_att, logvar_att)

            ##############################################
            # Image Reconstruction 
            ##############################################

            img_from_img = self.decoder['resnet_features'](z_from_img)
            img_from_att = self.decoder['resnet_features'](z_from_att)

            ##############################################
            # Attributes Reconstruction
            ##############################################
            
            att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
            att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
            
            ##############################################
            # Reconstruction Loss
            ##############################################
            
            reconstruction_img_loss = self.L1(img_from_img, img)
            reconstruction_att_loss = self.L1(att_from_att, att)
            reconstruction_loss = reconstruction_img_loss + reconstruction_att_loss
            
            ##############################################
            # Cross-reconstruction Loss
            ##############################################
            
            cross_reconstruction_img_loss = self.L1(img_from_att, img)
            cross_reconstruction_att_loss = self.L1(att_from_img, att)
            cross_reconstruction_loss = cross_reconstruction_img_loss + cross_reconstruction_att_loss

            ##############################################
            # Generator Loss using Discriminator 
            ##############################################

            pred_dis_img_from_img = self.net_D_Img(torch.cat((img_from_img, att), 1))
            pred_dis_img_from_att = self.net_D_Img(torch.cat((img_from_att, att), 1))

            pred_dis_att_from_att = self.net_D_Att(torch.cat((img, att_from_att), 1))
            pred_dis_att_from_img = self.net_D_Att(torch.cat((img, att_from_img), 1))

            real_label = utils.cuda(Variable(torch.ones(pred_dis_img_from_img.size())), self.device)

            gen_loss = self.MSE(pred_dis_img_from_img, real_label)
            gen_loss += self.MSE(pred_dis_img_from_att, real_label)

            gen_loss += self.MSE(pred_dis_att_from_att, real_label)
            gen_loss += self.MSE(pred_dis_att_from_img, real_label)

            ##############################################
            # KL-Divergence
            ##############################################

            KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
                    + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

            ##############################################
            # Distribution Alignment
            ##############################################
            distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                                    torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

            distance = distance.sum()

            ##############################################
            # scale the loss terms according to the warmup
            # schedule
            ##############################################

            f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
            f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
            cross_reconstruction_factor = torch.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])]).to(self.device)

            f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
            f2 = f2 * (1.0 * self.warmup['beta']['factor'])
            beta = torch.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])]).to(self.device)

            f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
            f3 = f3*(1.0*self.warmup['distance']['factor'])
            distance_factor = torch.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])]).to(self.device)

            ##############################################
            # Put the loss together and call the optimizer
            ##############################################

            loss_G = gen_loss - beta * KLD + reconstruction_loss

            if cross_reconstruction_loss>0:
                cross_reconstruction_loss = cross_reconstruction_factor*cross_reconstruction_loss
                loss_G += cross_reconstruction_loss
            if distance_factor >0:
                distance = distance_factor*distance
                loss_G += distance

            #Update Generator
            loss_G.backward()
            self.optimizer_G.step()


        ##############################################
        # Training the Discriminator Network
        ##############################################

        for dis_iter in range(0, self.num_dis_iter):
            utils.set_grad([self.net_D_Att, self.net_D_Img], True)
            self.optimizer_D.zero_grad()

            ##############################################
            # Sample from history of reconstructed data
            ##############################################

            att_from_att = Variable(torch.Tensor(self.att_fake_from_att_sample([att_from_att.cpu().data.numpy()])[0]))
            att_from_img = Variable(torch.Tensor(self.att_fake_from_img_sample([att_from_img.cpu().data.numpy()])[0]))
            img_from_img = Variable(torch.Tensor(self.img_fake_from_img_sample([img_from_img.cpu().data.numpy()])[0]))
            img_from_att = Variable(torch.Tensor(self.img_fake_from_att_sample([img_from_att.cpu().data.numpy()])[0]))
            att_from_att, att_from_img, img_from_img, img_from_att = utils.cuda([att_from_att, att_from_img, img_from_img, img_from_att], (self.device))

            ##############################################
            # Forward through Discriminator Network
            ##############################################

            #Attribute Discriminator
            pred_dis_att_real = self.net_D_Att(torch.cat((img, att), 1))
            pred_dis_att_from_att = self.net_D_Att(torch.cat((img, att_from_att), 1))
            pred_dis_att_from_img = self.net_D_Att(torch.cat((img, att_from_img), 1))

            real_label = utils.cuda(Variable(torch.ones(pred_dis_att_real.size())), (self.device))
            fake_label = utils.cuda(Variable(torch.zeros(pred_dis_att_from_att.size())), (self.device))

            att_real_loss = self.MSE(pred_dis_att_real, real_label)
            att_from_att_loss = self.MSE(pred_dis_att_from_att, fake_label)
            att_from_img_loss = self.MSE(pred_dis_att_from_img, fake_label)

            #Image Discriminator
            pred_dis_img_real = self.net_D_Img(torch.cat((img,att), 1))
            pred_dis_img_from_img = self.net_D_Img(torch.cat((img_from_img, att), 1))
            pred_dis_img_from_att = self.net_D_Img(torch.cat((img_from_att, att), 1))

            img_real_loss = self.MSE(pred_dis_img_real, real_label)
            img_from_img_loss = self.MSE(pred_dis_img_from_img, fake_label)
            img_from_att_loss = self.MSE(pred_dis_img_from_att, fake_label)

            #Total discriminator loss
            loss_D_att = (att_real_loss + att_from_att_loss + att_from_img_loss) / 3
            loss_D_img = (img_real_loss + img_from_img_loss + img_from_att_loss) / 3

            #Update discriminator
            loss_D_att.backward()
            loss_D_img.backward()
            self.optimizer_D.step()

        return loss_G.item(), loss_D_att.item(), loss_D_img.item(), \
                reconstruction_img_loss.clone().detach().item(), reconstruction_att_loss.clone().detach().item(), \
                cross_reconstruction_img_loss.clone().detach().item(), cross_reconstruction_att_loss.clone().detach().item(), \
                reconstruction_loss.clone().detach().item(), cross_reconstruction_loss.clone().detach().item(), \
                (beta*KLD).clone().detach().item(), (distance_factor*distance).clone().detach().item(), gen_loss.clone().detach().item()

    def train_vae(self, epoch):

        losses_G = []
        losses_D_att = []
        losses_D_img = []
        losses_log = []

#        self.dataloader = data.DataLoader(self.dataset,batch_size= self.batch_size,shuffle= True,drop_last=True)#,num_workers = 4)

        self.dataset.novelclasses =self.dataset.novelclasses.long().to(self.device)
        self.dataset.seenclasses =self.dataset.seenclasses.long().to(self.device)
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
#        for epoch in range(0, self.nepoch ):
        self.current_epoch = epoch

        i=-1

        total_loss_G = 0
        total_loss_D_att = 0
        total_loss_D_img = 0
        
        total_recon_img_loss = 0
        total_recon_att_loss = 0
        total_cross_recon_img_loss = 0
        total_cross_recon_att_loss = 0

        total_KLD = 0
        total_distance = 0
        total_DIS_loss = 0

        for iters in range(0, self.dataset.ntrain, self.batch_size):
            i+=1

            label, data_from_modalities = self.dataset.next_batch(self.batch_size)

            label= label.long().to(self.device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(self.device)
                data_from_modalities[j].requires_grad = False

            loss_G, loss_D_att, loss_D_img, recon_img_loss, recon_att_loss, cross_recon_img_loss, cross_recon_att_loss, \
            reconstruction_loss, cross_reconstruction_loss, \
            KLD, distance_loss, dis_loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], epoch, i)

            loss_log = ('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                        ' | recon_loss ' + str(reconstruction_loss)[:5] +
                        ' | cross_recon_loss ' + str(cross_reconstruction_loss)[:5] +
                        ' | Distance ' + str(distance_loss)[:5] +
                        ' | Gen_loss ' + str(dis_loss)[:5] + 
                        ' | loss_VAE ' +  str(loss_G)[:5] +
                        ' | loss_D_att ' +  str(loss_D_att)[:5] + 
                        ' | loss_D_img ' +  str(loss_D_img)[:5])

            total_loss_G += loss_G
            total_loss_D_att += loss_D_att
            total_loss_D_img += loss_D_img
            
            total_recon_img_loss += recon_img_loss
            total_recon_att_loss += recon_att_loss
            total_cross_recon_img_loss += cross_recon_img_loss
            total_cross_recon_att_loss += cross_recon_att_loss

            total_KLD += KLD
            total_distance += distance_loss
            total_DIS_loss += dis_loss

            if i%int(self.batch_size)==0:

                print(loss_log)

            if i%int(self.batch_size)==0 and i>0:
                losses_log.append(loss_log)
                losses_G.append(loss_G)
                losses_D_att.append(loss_D_att)
                losses_D_img.append(loss_D_img)

        # Show the losses graph using tensorboardx
        self.writer.add_scalar('Loss/loss_G', total_loss_G/self.dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/loss_D_Attribute', total_loss_D_att/self.dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/loss_D_Image', total_loss_D_img/self.dataset.ntrain, epoch)
        
        self.writer.add_scalar('Loss/Recon_Image_Loss', total_recon_img_loss/self .dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/Recon_Attribute_Loss', total_recon_att_loss/self.dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/Cross-recon_Image_Loss', total_cross_recon_img_loss/self.dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/Cross-recon_Attribute_Loss', total_cross_recon_att_loss/self.dataset.ntrain, epoch)

        self.writer.add_scalar('Loss/KLD_Loss', total_KLD/self .dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/Distance_Loss', total_distance/self .dataset.ntrain, epoch)
        self.writer.add_scalar('Loss/GEN_loss_from_DIS', total_DIS_loss/self .dataset.ntrain, epoch)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses_G, losses_D_att, losses_D_img, losses_log


    def train_classifier(self, show_plots=False):

        if self.num_shots > 0 :
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies


        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses


        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)


        self.clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).to(self.device).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.FloatTensor([]).to(self.device), torch.LongTensor([]).to(self.device)


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )

            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    novelclass_aux_data,
                    novel_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.FloatTensor([]).to(self.device)

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_img, z_unseen_img ,z_seen_att    ,z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)


        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################

        cls = classifier.CLASSIFIER(self.clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)

        #print(self.clf.state_dict())

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()

                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                self.writer.add_scalar('Accuracy/Seen_Acc', cls.acc_seen.clone().detach().item(), k)
                self.writer.add_scalar('Accuracy/Unseen_Acc', cls.acc_novel.clone().detach().item(), k)
                self.writer.add_scalar('Accuracy/Harmonic_Mean', cls.H.clone().detach().item(), k)

                history.append([cls.acc_seen.clone().detach().item(), cls.acc_novel.clone().detach().item(),
                                cls.H.clone().detach().item()])

            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, cls.acc.clone().detach().item(), 0])

        #cls.val_gzsl_class(test_seen_X, test_seen_Y, cls_seenclasses)
        #cls.val_gzsl_class(test_novel_X, test_novel_Y, cls_novelclasses)

        #print(self.clf.state_dict())

        if self.generalized:
            # print('\nbest_novel=%.4f, best_seen=%.4f, best_h=%.4f , loss=%.4f \n' % (
            # u.clone().detach().item(), s.clone().detach().item(), best_H.clone().detach().item(), average_loss))

            return cls.acc_novel.clone().detach().item(), cls.acc_seen.clone().detach().item(), cls.H.clone().detach().item(), history
        else:
            return 0, cls.acc.clone().detach().item(), 0, history
