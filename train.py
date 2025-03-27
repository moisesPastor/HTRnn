#!/usr/bin/env python3 # Standard packages 
import time 
import sys, os, argparse 
from tqdm import tqdm 
import torch 
import numpy as np 
from torchvision.utils import save_image 
from multiprocessing import cpu_count 
from termcolor import colored 
from torch.utils.data import Dataset, DataLoader #paralel 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce
import torch.distributed as dist
import torchvision.transforms as transforms

import fastwer
import re


# Local packages
import procImg
from buildMod import HTRModel
from dataset import HTRDataset, ctc_collate

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            output_model_path,
            batch_size: int,
    ) -> None:
        self.gpu_id = dist.get_rank();
        self.batch_size= batch_size

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id], broadcast_buffers=False)

        self.output_model_path = output_model_path
#        print("\nmodel loaded on gpu %d"% self.gpu_id)

        CTC_BLANK=1
        self.criterion = torch.nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epochs:int, early_stop:int):
       last_best_val_epoch=-1
       best_val_cer=sys.float_info.max;
       best_val_loss=sys.float_info.max;

       for epoch in range(epochs):
           device = torch.device(f'cuda:{self.gpu_id}') 
           total_train_loss = torch.tensor(0.0, device=device)
           total_val_loss   = torch.tensor(0.0, device=device)
           cer_score_train  = torch.tensor(0.0, device=device)
           wer_score_train  = torch.tensor(0.0, device=device)
           cer_score_val    = torch.tensor(0.0, device=device)
           wer_score_val    = torch.tensor(0.0, device=device)

           ignored_batches=list()
           batch_num=0

           try:
               terminal_cols = os.get_terminal_size()[0]
           except IOError:
               terminal_cols = 80
              
           format='{l_bar}{bar:'+str(terminal_cols-105)+'}{r_bar}'
            
           self.model.train()        
           train_str="Epoch "+str(epoch)+" gpu "+str(self.gpu_id)
           with tqdm(self.train_loader, bar_format=format,dynamic_ncols=True, colour='green', desc=train_str,position=self.gpu_id) as tq:
              for ((x,  input_lengths),(y,target_lengths), bIdxs) in tq:

                  x = x.to(self.gpu_id)
                  y = y.to(self.gpu_id)

#                  save_image(x, f"out.png", nrow=1); sys.exit(1)

                  try:
                      self.optimizer.zero_grad()
                      outputs = self.model(x)
                      
                      loss = self.criterion(outputs, y, input_lengths=input_lengths, target_lengths=target_lengths)
                      loss.backward()
                      self.optimizer.step()

                      total_train_loss += loss.item()
                      cer, wer = self.get_cer_wer(outputs,y,target_lengths)
                      cer_score_train+=cer
                      wer_score_train+=wer
                      tq.set_postfix({'Train loss': loss.item(), 'Train cer': cer, 'Train wer':wer})
                     
                  except OverflowError:
                      print("OverFlowError")

                  except MemoryError:
                      print("Memory Error")

                  except NotImplementedError as err:
                      print("Non Implemented Error")

                  except RuntimeError as err:
                      print(str(err))
                      sys.exit(-1)
                      continue

                  except Exception as e:
                      print(e)
                      torch.cuda.empty_cache()
                      continue

           self.model.eval()    
        
           with torch.no_grad():
             #total_val_loss = 0
             val_str="Valid  "+str(epoch)+" gpu "+str(self.gpu_id)
             tq_val = tqdm(self.val_loader, bar_format=format, colour='magenta', desc=val_str,position=self.gpu_id)
             for ((x, input_lengths),(y,target_lengths), bIdxs) in tq_val:
#                 torch.cuda.empty_cache()
                 try:
                     x = x.to(device)
                     #save_image(x, f"out.png", nrow=1); break
                     
                     outputs = self.model.forward(x)
                     # outputs ---> W,N,K    K=number of different chars + 1

                     loss =  self.criterion(outputs, y, input_lengths=input_lengths,target_lengths=target_lengths)
                     total_val_loss += loss.item()
                 except Exception as er:
                     #if verbosity:                 
                     #print("ERROR: CUDA out of memory\n")
                     print(er)

                     continue
                 cer_val, wer_val = self.get_cer_wer(outputs,y,target_lengths)
                 cer_score_val += cer_val
                 wer_score_val += wer_val
                 tq_val.set_postfix({'Train loss': loss.item(),'cer': cer_val, 'wer': wer_val})


           dist.reduce(total_train_loss, dst=0)
           dist.reduce(total_val_loss, dst=0)
           dist.reduce(cer_score_train, dst=0)
           dist.reduce(wer_score_train, dst=0)
           dist.reduce(cer_score_val, dst=0)
           dist.reduce(wer_score_val, dst=0)


           if self.gpu_id == 0:
               train_loss_mean = total_train_loss / len(self.train_loader) / dist.get_world_size()
               val_loss_mean = total_val_loss / len(self.val_loader) / dist.get_world_size()
               cer_tr = cer_score_train/len(self.train_loader) / dist.get_world_size() # - len(ignored_batches) )
               wer_tr = wer_score_train/len(self.train_loader) / dist.get_world_size() # - len(ignored_batches) )
               cer_vl = cer_score_val/(len(self.val_loader)) / dist.get_world_size()
               wer_vl = wer_score_val/(len(self.val_loader)) / dist.get_world_size()

               num_img_processed = len(self.train_loader) * self.batch_size #len(bIdxs))
               if (val_loss_mean  < best_val_loss):   #if (cer_vl  < best_val_cer):

                 print ("\033[93m\n\ttrain av. loss = %.5f val av. loss = %.5f train cer = %.2f train wer = %.2f val cer = %.2f val wer = %.2f\x1b[0m"%(train_loss_mean, val_loss_mean,cer_tr, wer_tr, cer_vl, wer_vl))

                 epochs_without_improving=0
                 best_val_loss = val_loss_mean.item()
                 #best_val_cer = cer_vl

                 torch.save({'model': self.model.module, 
                 'optimizer': self.optimizer.state_dict(),
                 'codec': self.train_loader.dataset.char_voc,
                 'spaceSymbol': self.train_loader.dataset.spaceChar}, self.output_model_path+"_"+str(epoch)+".pth")
#                 'line_height': args.fixed_height,

                 if os.path.exists(self.output_model_path+"_"+str(last_best_val_epoch)+".pth"):
                      os.remove(self.output_model_path+"_"+str(last_best_val_epoch)+".pth")
                 last_best_val_epoch=epoch

               else:
                 epochs_without_improving = epochs_without_improving + 1;
                 print ("\n\ttrain av. loss = %.5f val av. loss = %.5f train cer = %.2f train wer = %.2f val cer = %.2f val wer = %.2f"%(train_loss_mean,val_loss_mean,cer_tr, wer_tr, cer_vl, wer_vl))

               if epochs_without_improving >= early_stop:
                    destroy_process_group()
                    sys.exit(colored("Early stoped after %i epoch without improving"%(early_stop),"green"))





    def ctc_remove_successives_identical_ind(ind):
       ind=re.sub("(<ctc>)+", '·',ind).strip(" ")

       res = ""
       for i in range(len(ind)):
           if len(res) > 0 and res[len(res)-1] == ind[i]:
                continue
           res+=(ind[i])

       res=re.sub('·',"", res).strip(" ")
       return res

    def get_cer_wer(self, outputs,y,target_lengths):
       with torch.no_grad():
         ref, hyp = list(), list()
         ptr=0;
         outputs = outputs.permute(1, 0, 2)
         output = torch.argmax(outputs, dim=2).cpu().numpy()

         for i, batch in enumerate(output):
               yi = y[ptr:ptr+target_lengths[i]]
               refText = self.train_loader.dataset.get_decoded_label(yi.cpu().numpy())
               refText=refText.replace("( )+", ' ').strip(" ")
               ref.append(refText)
               ptr += target_lengths[i]

               hypText = self.train_loader.dataset.get_decoded_label(batch)
               hypText = self.ctc_remove_successives_identical_ind(hypText)
               hypText=hypText.replace("( )+", ' ').strip(" ")

               hyp.append(hypText)
               '''
               if len(hypText) > 0:
                   print("\nRef: "+refText)
                   print("Hyp: "+hypText)
                   print("Cer: "+str(fastwer.score([hypText],[refText], char_level=True)))
               '''
       cer=fastwer.score(hyp,ref,char_level=True)
       wer=fastwer.score(hyp,ref,char_level=False)
       return cer,wer
    
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]= "localhost"
    os.environ["MASTER_PORT"]= "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #,find_unused_parameters=True)


def get_data_loaders(dataset_train, dataset_val, space_symbol, charVoc, batch_size):
    htr_dataset_train = HTRDataset(dataset_train,
                                   space_symbol,
                                   charVoc=charVoc)
#                                   transform=img_transforms)

    htr_dataset_val = HTRDataset(dataset_val,
                                 space_symbol,
                                 charVoc=charVoc)
 #                                transform=img_transforms)
    
    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores

    train_loader = torch.utils.data.DataLoader(htr_dataset_train,
                                               batch_size = batch_size,
                                               num_workers=nwrks,
                                               pin_memory=True,
                                               shuffle = False, 
                                               sampler = DistributedSampler(htr_dataset_train),
                                               collate_fn = ctc_collate)

 
    val_loader = torch.utils.data.DataLoader(htr_dataset_val,
                                             batch_size = batch_size,
                                             num_workers=nwrks,
                                             pin_memory=True,
                                             shuffle = False, 
                                             collate_fn = ctc_collate)

    return train_loader, val_loader

def load_model(model_name, models_file, fixed_height:int,rank:int):

    if os.path.isfile(model_name):        
        
        state = torch.load(model_name)# ,map_location=torch.device(rank))
        model = state['model']
        charVoc = np.array(state['codec'])
        
        #print(state.items())
        #print(model)
    else:
         charVoc = np.array([])
         try:
            file = open(models_file)   

            lines = file.read().splitlines()
            for char_name in lines:
                if len(char_name) >= 1:                
                    charVoc = np.append(charVoc,char_name);

         except FileNotFoundError:
            print(colored("\tWARNING: file  "+ models_file + " does not exist","red"))
            exit (-1)
        
         model = HTRModel(num_classes=len(charVoc),line_height=fixed_height)

#    for name, param in model.named_parameters():
#        if param.requires_grad:
#            print (name, param.is_cuda, param.get_device())

    return model, charVoc

#def main(rank:int, world_size:int, model_path, model_names_file, dataset_train, dataset_val, space_symbol, batch_size, fixed_height:int, epochs:int):
def main(rank:int, args):
    world_size=torch.cuda.device_count()
    ddp_setup(rank,world_size)

    model, charVoc = load_model(args.model_name, args.models_file, args.fixed_height,rank)

    train_loader, val_loader = get_data_loaders(args.dataset_train, args.dataset_val, args.space_symbol, charVoc, args.batch_size)

    output_model_path = os.path.splitext(args.model_name)[0]
    t = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, output_model_path=output_model_path, batch_size=args.batch_size)
    t.train(args.epochs, early_stop=args.early_stop);

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training using the given dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models_file', type=str, help='file with models name',required=False)
    parser.add_argument('--data_augm', action='store_true', help='enable data augmentation', default=False)
    parser.add_argument('--fixed_height', type=int, help='fixed image height', default=64)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--early_stop', type=int, help='number of epochs without improving', default=10)
    parser.add_argument('--batch_size', type=int, help='image batch-size', default=24)
    parser.add_argument('--space_symbol', type=str, help='image batch-size', default='~')
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')     
    parser.add_argument("--verbosity", action="store_true",  help="increase output verbosity",default=False)
    parser.add_argument('dataset_train', type=str, help='train dataset location')
    parser.add_argument('dataset_val', type=str, help='validation dataset location')
    parser.add_argument('model_name', type=str, help='Save model with this file name')
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")


    world_size=torch.cuda.device_count()
    mp.spawn(main, args = (args,), nprocs=world_size, join=True)
    #mp.spawn(main, args=(world_size, args.model_name, args.models_file, args.dataset_train, args.dataset_val, args.space_symbol, args.batch_size, args.fixed_height, args.epochs), nprocs=world_size)

    sys.exit(os.EX_OK)

