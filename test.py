#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import torch
import fastwer
import numpy as np
import re

#from torchvision.utils import save_image
from multiprocessing import cpu_count
from tqdm import tqdm

#distributed packages
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce
import torch.distributed as dist

# Local packages
import procImg
from dataset import HTRDataset, ctc_collate
from ctcdecode import CTCBeamDecoder

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]= "localhost"
    os.environ["MASTER_PORT"]= "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #,find_unused_parameters=True)

def ctc_remove_successives_identical_ind(ind):
    res = ""
    for i in range(len(ind)):
        if len(res) > 0 and res[len(res)-1] == ind[i]:
             continue
        res+=(ind[i])

    res=re.sub("(<ctc>)+", '',res).strip(" ")
    return res

def test(model, htr_dataset, test_loader, rank:int, bs=20, lmFile=None, beam=200, gsf=1.0, wip=0.0, verbosity=True):
    
    CTC_BLANK = 1
    
    if (verbosity):
        charVoc=htr_dataset.get_charVoc();
        print("models= %s %i models"%(htr_dataset.get_charVoc(),htr_dataset.get_num_classes()))
        print("CTC_BLANK= %s"%(charVoc[CTC_BLANK]))
        print('Space symbol= \"%s\"'%(htr_dataset.get_spaceChar()))
    
    # There are some specific layers/parts of the model that behave
    # differently during training and evaluation time (Dropouts,
    # BatchNorm, etc.). To turn off them during model evaluation:
    model.eval()    
    
    # Deactivate the autograd engine
    # It's not required for inference phase
    with torch.no_grad():

        # https://github.com/parlance/ctcdecode
        # In this case, parameters alpha and beta:
        #     alpha --> grammar scale factor
        #      beta --> word insertion penalty
        if  lmFile is not None:
            decoder = CTCBeamDecoder(htr_dataset.get_charVoc(),
                                model_path=lmFile,
                                alpha=gsf if lmFile else 0.0,
                                beta=wip if lmFile else 0.0,
                                beam_width=beam,
                                #cutoff_top_n=200,
                                #cutoff_prob=1.0,
                                num_processes=int(cpu_count() * 0.8),
                                blank_id=CTC_BLANK,
                                log_probs_input=True)
        

        # To store the reference and hypothesis strings
        ref, hyp = list(), list()
        
        train_str="Test gpu "+str(rank)
        if (verbosity):
            tq=tqdm(test_loader,colour='cyan', desc=train_str, position=rank, disable=True)
        else:
            tq=tqdm(test_loader,colour='cyan', desc=train_str, position=rank)

        for ((x, input_lengths),(y,target_lengths), bIdxs) in tq:
            x = x.to(dist.get_rank())
            #save_image(x, f"out.png", nrow=1); break

            # Run forward pass (equivalent to: outputs = model(x))
            outputs = model.forward(x)
            # outputs ---> W,N,K    K=number of different chars + 1

            outputs = outputs.permute(1, 0, 2)
            # outputs ---> N,W,K    (BATCHSIZE, #TIMESTEPS, #LABELS)

            if lmFile is None:
                output = torch.argmax(outputs, dim=2).cpu().numpy()
            else:
                output, scores, ts, out_seq_len = decoder.decode(outputs.data, torch.IntTensor(input_lengths))
                # output ---> N,N_BEAMS,N_TIMESTEPS   
                #             default: N_BEAMS=100 (beam_width)
            
            # Decode outputted tensors into text lines
            #assert len(output) == len(target_lengths)
           
            ptr = 0
            for i, batch in enumerate(output):
                if lmFile is not None:
                   batch, size =  batch[0], out_seq_len[i][0]

                yi = y[ptr:ptr+target_lengths[i]]

                refText = htr_dataset.get_decoded_label(yi)
                refText=refText.replace("( )+", ' ').strip(" ")
                ref.append(refText)

                ptr += target_lengths[i]
                if lmFile is not None:
                    hypText = htr_dataset.get_decoded_label(batch[0:size]) if size > 0 else ''
                else:
                    hypText = htr_dataset.get_decoded_label(batch) 

                hypText = ctc_remove_successives_identical_ind(hypText)
                hypText=hypText.replace("( )+", ' ').strip(" ")
                hyp.append(hypText)
                
                if (verbosity):
                    cer_score=fastwer.score([hypText], [refText], char_level=True)
                    line_id = htr_dataset.items[bIdxs[i]]
                    if (cer_score == 0):
                        print("\033[93m Line-ID: %s\n\033[92mREF: \"%s\"\nHYP: \"%s\"\033[39m\n"%(line_id,refText,hypText))
                    else:
                        wer_score = fastwer.score([hypText], [refText], char_level=False)
                        print("\033[93mLine-ID: %s \033[31mcer=%.3f wer=%.3f\033[39m"%(line_id,cer_score,wer_score))
                        print("REF: \"%s\"\nHYP: \"%s\"\n"%(refText,hypText))
                        

    cer = fastwer.score(hyp, ref, char_level=True)
    wer = fastwer.score(hyp, ref, char_level=False)

    device = torch.device(f'cuda:{rank}')
    cer = torch.tensor(cer, device=device)
    wer = torch.tensor(wer, device=device)

    dist.reduce(cer, dst=0)
    dist.reduce(wer, dst=0)

    if rank == 0:
       # Compute the total number of parameters of the trained model
       total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       cer = cer / dist.get_world_size() 
       wer = wer / dist.get_world_size() 

       print('\n'+'#'*30, file=sys.stderr)
       print(f'# Model\'s params: {total_params}', file=sys.stderr)
       print("             CER: {:5.2f}%".format(cer), file=sys.stderr)
       print("             WER: {:5.2f}%".format(wer), file=sys.stderr)
       print('#'*30, file=sys.stderr)


def main(rank:int, args):
    world_size=torch.cuda.device_count()
    ddp_setup(rank,world_size)

    # Load model in memory
    state = torch.load(args.model)#, map_location=device)
    model = state['model']
    model = model.to(dist.get_rank())
    model = DDP(model, device_ids=[dist.get_rank()], broadcast_buffers=False)
   
    # Get the sequence of transformations to apply to images 
    #img_transforms = procImg.get_tranform(state['line_height'])
          
    htr_dataset = HTRDataset(args.dataset,
                            state['spaceSymbol'],
    #                        transform=img_transforms,
                            charVoc=state['codec'])

    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    test_loader = torch.utils.data.DataLoader(htr_dataset,
                                            batch_size = args.batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = False,
                                            sampler = DistributedSampler(htr_dataset),
                                            collate_fn = ctc_collate)


    test(model, htr_dataset, test_loader, 
         rank=rank,
         bs=args.batch_size,
         lmFile=args.lm_model,
         beam=args.beam_width,
         gsf=args.gsf,
         wip=args.wip,
         verbosity=args.verbosity)

    destroy_process_group()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing a HTR model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lm-model', type=str, help='laguage model file (Arpa/Kenlm format)', default=None)
    parser.add_argument('--beam-width', type=int, help='this sets up how broad the beam search is', default=200)
    parser.add_argument('--gsf', type=float, help='this sets up Grammar scale factor', default=1.0)
    parser.add_argument('--wip', type=float, help='this sets up Word insertion penalty', default=0.0)
    parser.add_argument('--batch_size', type=int, help='image batch-size', default=24)
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--verbosity", action="store_true",  help="increase output verbosity",default=False) 
    parser.add_argument('model', type=str, help='PyTorch model file')
    parser.add_argument('dataset', type=str, help='dataset path')

    args = parser.parse_args()
    print ("\n"+str(sys.argv) )
    
    world_size=torch.cuda.device_count()
    mp.spawn(main, args = (args,), nprocs=world_size, join=True)
  
    
    sys.exit(os.EX_OK)
