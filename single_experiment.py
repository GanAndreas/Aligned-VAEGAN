
### execute this function to train and test the vae-model

from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import argparse
import datetime

print('\n' + str(datetime.datetime.now()) + '\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
#parser.add_argument('--num_shots',type=int)
#parser.add_argument('--generalized', type=str2bool)
parser.add_argument('--pretrain', type=str2bool)
parser.add_argument('--mod_dataset')
parser.add_argument('--device')
parser.add_argument('--num_gen_iter', type=int, default=1)
parser.add_argument('--num_dis_iter', type=int, default=1)
args = parser.parse_args()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.13,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': 0.00001,
    'generalized': True,
    'pretrain': False,
    'mod_dataset': '',
    'num_gen_iter': 5,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 300,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660),
                        'attributes': (1450, 665),
                        'sentences': (1450, 665) },
    'latent_size': 64
}


# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 31},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
      #changed for ablation {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 36},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 27},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 31},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 100},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78}
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
#hyperparameters['num_shots']= args.num_shots
#hyperparameters['generalized']= args.generalized
hyperparameters['pretrain'] = args.pretrain
hyperparameters['mod_dataset'] = args.mod_dataset
hyperparameters['device']= args.device
hyperparameters['num_gen_iter']= args.num_gen_iter
hyperparameters['num_dis_iter']= args.num_dis_iter

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                                'APY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                                'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
                                                    'APY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
                                                    'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 0), 'SUN': (0, 0, 200, 0),
                                                    'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 200, 0),
                                                    'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
                                                    'APY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
                                                    'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200)}


if (hyperparameters['dataset'] == 'CUB'):
    hyperparameters['latent_size'] = 128
# if (hyperparameters['dataset'] == 'SUN'):
#     hyperparameters['latent_size'] = 128

model = Model( hyperparameters)
model.to(hyperparameters['device'])

losses_log = 0

if hyperparameters['pretrain']:

    ########################################
    ### load model where u left
    ########################################

    # load VAEGAN parameters from modified model
    
    if hyperparameters['mod_dataset']:
        saved_best_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_BestEp.pth.tar')
        best_epoch = saved_best_state['best_epoch']

        saved_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(best_epoch) + '.pth.tar')
        print('\n USING SAVED PRETRAIN PARAMETER \n CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(best_epoch) + '.pth.tar \n')

        for d in model.all_data_sources:
            model.encoder[d].load_state_dict(saved_state['encoder'][d])
            model.decoder[d].load_state_dict(saved_state['decoder'][d])

    # load VAEGAN parameters from original model

    else:
        saved_best_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_BestEp.pth.tar')
        best_epoch = saved_best_state['best_epoch']

        saved_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_ep' + str(best_epoch) + '.pth.tar')
        print('\n USING SAVED PRETRAIN PARAMETER \n CADA_trained_' + hyperparameters['dataset'] + '_ep' + str(best_epoch) + '.pth.tar \n')

        for d in model.all_data_sources:
            model.encoder[d].load_state_dict(saved_state['encoder'][d])
            model.decoder[d].load_state_dict(saved_state['decoder'][d])

    ########################################

    # train the classifier using the loaded VAEGAN parameters

    model.clf.eval()
    u,s,h,history = model.train_classifier()

# train VAEGAN model without pretrained parameters

else:
    best_u = 0.0
    best_s = 0.0
    best_h = -1.0
    history = [0.0, 0.0, 0.0, 0.0]
    best_epoch = 0


    for epoch in range(0, hyperparameters['epochs']):
        losses_G, losses_D_att, losses_D_img, losses_log = model.train_vae(epoch)
        
        # train the classifier 
        if (epoch%5==0 and epoch >= 20) or (epoch==(hyperparameters['epochs']-1)):

            u,s,h,history = model.train_classifier()
            
            if h > best_h:
                best_u = u
                best_s = s
                best_h = h
                best_epoch = epoch

                print('\nbest epoch :' + str(best_epoch) + '\n')
                print(model.encoder['resnet_features'].state_dict()['feature_encoder.0.weight'])
                
            state = {
                    'model': model.state_dict(),
                    'hyperparameters':hyperparameters,
                    'epoch': epoch,
                    'encoder':{'resnet_features': model.encoder['resnet_features'].state_dict(),
                               'attributes': model.encoder['attributes'].state_dict()
                              },
                    'decoder':{'resnet_features': model.decoder['resnet_features'].state_dict(),
                               'attributes': model.decoder['attributes'].state_dict()
                              },
                    'discriminator': {'net_D_Att': model.net_D_Att.state_dict(),
                                      'net_D_Img': model.net_D_Img.state_dict()
                                     },
                    'optimizer_G': model.optimizer_G.state_dict(),
                    'optimized_D': model.optimizer_D.state_dict(),
                    'loss_log': losses_log
                    }
            
            # saving VAEGAN parameters for each epoch
            if hyperparameters['mod_dataset']:
                torch.save(state, 'param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(epoch) + '.pth.tar')
                print('>> saved CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(epoch) + '.pth.tar')
            else:
                torch.save(state, 'param/CADA_trained_' + hyperparameters['dataset'] + '_ep' + str(epoch) + '.pth.tar')
                print('>> saved CADA_trained_' + hyperparameters['dataset'] + '_ep' + str(epoch) + '.pth.tar')

            print('\nBest VAE Epoch %.1f     | Novel %.4f | Seen %.4f | H %.4f \n' % (
            best_epoch, u, s, best_h))

    # checking the saved parameters
    best_state = {'best_epoch': best_epoch}
    if hyperparameters['mod_dataset']:
        torch.save(best_state, 'param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_BestEp.pth.tar')
        print('>> saved CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_BestEp.pth.tar')
    else:
        torch.save(best_state, 'param/CADA_trained_' + hyperparameters['dataset'] + '_BestEp.pth.tar')
        print('>> saved CADA_trained_' + hyperparameters['dataset'] + '_BestEp.pth.tar')

    hyperparameters['pretrain']=True
    model.pretrain = True
    print('\n\nCHECKING BEST CLASSIFIER MODEL PARAMETER')
    if hyperparameters['mod_dataset']:
        saved_best_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_BestEp.pth.tar')
        best_epoch = saved_best_state['best_epoch']

        saved_state = torch.load('./param/CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(best_epoch) + '.pth.tar')
        print('\nUSING SAVED PRETRAIN PARAMETER \n CADA_trained_' + hyperparameters['dataset'] + '_' + hyperparameters['mod_dataset'] + '_ep' + str(best_epoch) + '.pth.tar \n')


        for d in model.all_data_sources:
            model.encoder[d].load_state_dict(saved_state['encoder'][d])
            model.decoder[d].load_state_dict(saved_state['decoder'][d])

    print(model.encoder['resnet_features'].state_dict()['feature_encoder.0.weight'])




