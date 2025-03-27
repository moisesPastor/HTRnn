#!/usr/bin/env python3

# Standard packages
import time

import sys, os, argparse
from tqdm import tqdm
import torch
import numpy as np
#from torchvision.utils import save_image
from multiprocessing import cpu_count
from termcolor import colored
import editdistance
import fastwer
import re

# Local packages
import procImg
from buildMod import HTRModel
from dataset import HTRDataset, ctc_collate

def ctc_remove_successives_identical_ind(ind):

        ind=re.sub("(<ctc>)+", '',ind).strip(" ")

        res = ""
        for i in range(len(ind)):
            if len(res) > 0 and res[len(res)-1] == ind[i]:
                continue
            res+=(ind[i])

        return res

def get_cer_wer(outputs,y,target_lengths):
    with torch.no_grad():
         ref, hyp = list(), list()
         ptr=0;
         outputs = outputs.permute(1, 0, 2)
         output = torch.argmax(outputs, dim=2).cpu().numpy()

         for i, batch in enumerate(output):
               yi = y[ptr:ptr+target_lengths[i]]
               refText = htr_dataset_train.get_decoded_label(yi.cpu().numpy())
               refText=refText.replace("( )+", ' ').strip(" ")
               ref.append(refText)
               ptr += target_lengths[i]

               hypText = htr_dataset_train.get_decoded_label(batch)
               hypText = ctc_remove_successives_identical_ind(hypText)
               hypText=hypText.replace("( )+", ' ').strip(" ")

               hyp.append(hypText)
#                        if len(hypText) > 0:
#                            print("\nRef: "+refText)
#                            print("Hyp: "+hypText)
#                            print("Cer: "+str(fastwer.score([hypText],[refText], char_level=True)))
    cer=fastwer.score(hyp,ref,char_level=True)
    wer=fastwer.score(hyp,ref,char_level=False)
    return cer,wer


def train(model, htr_dataset_train ,htr_dataset_val, device, epochs=20, batch_size=24, early_stop=10,verbosity=False):
    # To control the reproducibility of the experiments
    #torch.manual_seed(17)
    charVoc = htr_dataset_train.get_charVoc()
    print("models= %s %i models"%(charVoc,htr_dataset_train.get_num_classes()))
    
    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    train_loader = torch.utils.data.DataLoader(htr_dataset_train,
                                            batch_size = batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = True, 
                                            collate_fn = ctc_collate)
    
    val_loader = torch.utils.data.DataLoader(htr_dataset_val,
                                            batch_size = batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = False, 
                                            collate_fn = ctc_collate)
    

    CTC_BLANK = 1
    print("CTC_BLANK= %s"%(charVoc[CTC_BLANK]))
    print('Space symbol= \"%s\"'%(htr_dataset_train.get_spaceChar()))
    # The Connectionist Temporal Classification loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    criterion = torch.nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    
    # Whether to zero infinite losses and the associated gradients.
    # Infinite losses mainly occur when the inputs are too short to be 
    # aligned to the targets.

    # https://pytorch.org/docs/stable/optim.html
    # Optimizers are algorithms or methods used to change the attributes of 
    # the neural network such as weights and learning rate to reduce the 
    # losses.
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    '''
    # Print model state_dict
    print("Model state_dict:",file=sys.stderr)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file=sys.stderr)
    # Print optimizer state_dict
    print("Optimizer state_dict:", file=sys.stderr)
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name], file=sys.stderr)
    '''
    # Print the total number of parameters of the model to train
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel with {total_params} parameters to be trained.\n', file=sys.stdout)

    # Epoch loop
    last_best_val_epoch=-1
    best_val_loss=sys.float_info.max;
    epochs_without_improving=0
    for epoch in range(epochs):
        total_train_loss = 0
        ignored_batches=list()
        batch_num=0
        cer_score_train=0
        wer_score_train=0
        cer_score_val=0
        wer_score_val=0
        model.train()

        try:
            terminal_cols = os.get_terminal_size()[0]
        except IOError:
            terminal_cols = 80
            
        format='{l_bar}{bar:'+str(terminal_cols-100)+'}{r_bar}'
            
        # Mini-Batch train loop
        print("Epoch %i"%(epoch))
        tq=tqdm(train_loader, bar_format=format,dynamic_ncols=True, colour='green', desc='  Train')

        for ((x,  input_lengths),(y,target_lengths), bIdxs) in tq:
            model.train()
            x, y = x.to(device), y.to(device)
           
            #save_image(x, f"out.png", nrow=1); sys.exit(1)
            
            try:
                outputs = model(x) # outputs ---> Tensor:TxNxC 
                # T=time-steps, N=batch-size, C=# of classes
            
                loss = criterion(outputs, y, input_lengths=input_lengths,       
                                 target_lengths=target_lengths)
           
                optimizer.zero_grad()
                
                # Compute dloss/dw for every parameter w which has
                # requires_grad=True: w.grad += dloss/dw
                loss.backward()
                
                # Optimizer method "step()" updates the parameters:
                # w += -lr * w.grad
                optimizer.step()

                total_train_loss += loss.item()
                cer,wer = get_cer_wer(outputs,y,target_lengths)
                cer_score_train+=cer
                wer_score_train+=wer
                tq.set_postfix({'Train loss': loss.item(),'cer': cer, 'wer': wer })

            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                ignored_batches.append(batch_num)
                if verbosity:
                    files=""
                    for i,x in enumerate(bIdxs):
                        files=files+" "+htr_dataset_train.items[bIdxs[i]]

                    print("     Ignored batch "+ str(batch_num)+" [" + files +" ]")
                continue

            batch_num=batch_num + 1

        if len(ignored_batches) > 0:
            print(colored("  Ignored batches "+str(ignored_batches)+" of "+str(batch_num),"green"))

        model.eval()    
        
        with torch.no_grad():
             val_loss = 0
             tq_val = tqdm(val_loader, bar_format=format, colour='magenta', desc='  Valid')
             for ((x, input_lengths),(y,target_lengths), bIdxs) in tq_val:
                 torch.cuda.empty_cache()
                 try:
                     x = x.to(device)
                     #save_image(x, f"out.png", nrow=1); break
                     
                     # Run forward pass (equivalent to: outputs = model(x))
                     outputs = model.forward(x)
                     # outputs ---> W,N,K    K=number of different chars + 1

                     loss =  criterion(outputs, y, input_lengths=input_lengths,       
                            target_lengths=target_lengths)
                     val_loss += loss.item()
                 except Exception as er:
                     if verbosity:                 
                         print("ERROR: CUDA out of memory\n")
                         print(er)

                     continue
                 cer_val, wer_val = get_cer_wer(outputs,y,target_lengths)
                 cer_score_val += cer_val
                 wer_score_val += wer_val
                 tq_val.set_postfix({'Train loss': loss.item(),'cer': cer_val, 'wer': wer_val})
            
        num_img_processed = len(train_loader) - (len(ignored_batches) * batch_size) #len(bIdxs))
        if num_img_processed == 0:
            print(colored("\ttrain av. loss = INF val av. loss = INF","red"))
            continue
      

        cer_tr = cer_score_train/(len(train_loader) - len(ignored_batches) )
        wer_tr = wer_score_train/(len(train_loader) - len(ignored_batches) )
        cer_vl = cer_score_val/(len(val_loader))
        wer_vl = wer_score_val/(len(val_loader))

        if (val_loss  < best_val_loss):
            print ("\033[93m\ttrain av. loss = %.5f val av. loss = %.5f train cer = %.2f train wer = %.2f val cer = %.2f val wer = %.2f\x1b[0m"%(total_train_loss/num_img_processed,val_loss/len(val_loader),cer_tr, wer_tr, cer_vl, wer_vl))

            logs.write("epoch %i train av. loss = %.5f val av. loss = %.5f train cer = %.2f val train wer = %.2f val cer = %.2f val wer = %.2f\n"%(epoch, total_train_loss/num_img_processed,val_loss/len(val_loader), cer_tr, wer_tr, cer_vl, wer_vl))
            logs.flush()
            
            epochs_without_improving=0
            best_val_loss = val_loss
            torch.save({'model': model, 
               'line_height': args.fixed_height, 
               'codec': charVoc,
               'spaceSymbol': args.space_symbol},
                      args.model_name.rsplit('.',1)[0]+"_"+str(epoch)+".pth")
            if os.path.exists(args.model_name.rsplit('.',1)[0]+"_"+str(last_best_val_epoch)+".pth"):
                os. remove(args.model_name.rsplit('.',1)[0]+"_"+str(last_best_val_epoch)+".pth")
            last_best_val_epoch=epoch

        else:
            epochs_without_improving = epochs_without_improving + 1;
            print ("\ttrain av. loss = %.5f val av. loss = %.5f train cer = %.2f train wer = %.2f val cer = %.2f val wer = %.2f"%(total_train_loss/num_img_processed,val_loss/len(val_loader),cer_tr, wer_tr, cer_vl, wer_vl))
            
        if epochs_without_improving >= early_stop:
            sys.exit(colored("Early stoped after %i epoch without improving"%(early_stop),"green"))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a model training process of using the given dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models-file', type=str, help='file with models name',required=False)
    parser.add_argument('--data-augm', action='store_true', help='enable data augmentation', default=False)
    parser.add_argument('--fixed-height', type=int, help='fixed image height', default=64)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--early-stop', type=int, help='number of epochs without improving', default=10)
    parser.add_argument('--batch-size', type=int, help='image batch-size', default=24)
    parser.add_argument('--space-symbol', type=str, help='image batch-size', default='~')
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')     
    parser.add_argument("--verbosity", action="store_true",  help="increase output verbosity",default=False)
    parser.add_argument('dataset_train', type=str, help='train dataset location')
    parser.add_argument('dataset_val', type=str, help='validation dataset location')
    parser.add_argument('model_name', type=str, help='Save model with this file name')
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DDDDDDDDDD")
    print(device)
    print("DDDDDDDDDD")
    
    if args.gpu:
        if args.gpu[0] > torch.cuda.device_count() or args.gpu[0] < 0:
            sys.exit(colored("\tERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()),"red"))
        torch.cuda.set_device(args.gpu[0])

    print("Selected GPU %i\n"%(torch.cuda.current_device()))
          
    if os.path.isfile(args.model_name):        
        state = torch.load(args.model_name, map_location=device)
        model = state['model']
        charVoc = np.array(state['codec'])
    else:
        charVoc = np.array([])
        try:
            file = open(args.models_file)   

            lines = file.read().splitlines()
            for char_name in lines:
                if len(char_name) >= 1:                
                    charVoc = np.append(charVoc,char_name);


        except FileNotFoundError:
            print(colored("\tWARNING: file  "+ args.models_file + " does not exist","red"))
            exit (-1)
        
        numClasses = len(charVoc)
        model = HTRModel(num_classes=numClasses,line_height=args.fixed_height)
        model.to(device)

        
    img_transforms = procImg.get_tranform(args.fixed_height, args.data_augm)

    htr_dataset_train = HTRDataset(args.dataset_train,
                                   args.space_symbol,
                                   transform=img_transforms,
                                   charVoc=charVoc)

    htr_dataset_val = HTRDataset(args.dataset_val,
                                 args.space_symbol,
                                 transform=img_transforms,
                                 charVoc=charVoc)

    start_time=time.time();
    logs = open(args.model_name.rsplit('.',1)[0]+".log","w")
    train(model, htr_dataset_train, htr_dataset_val, device, epochs=args.epochs,          
          batch_size=args.batch_size, early_stop=args.early_stop,
          verbosity=args.verbosity)

#    torch.save({'model': model, 
#                'line_height': args.fixed_height, 
#                'codec': charVoc,
#                'spaceSymbol': args.space_symbol}, args.model_name)


    logs.close()
    total_time=time.time() - start_time;
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    print("Total training time %02i:%02i:%02i"%(h,m,s))
    
    sys.exit(os.EX_OK)
