#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import torch
import fastwer
import numpy as np
import re
from cuda_selector import auto_cuda

#from torchvision.utils import save_image
from multiprocessing import cpu_count
from tqdm import tqdm

# Local packages
import procImg
from dataset import HTRDataset, ctc_collate
from ctcdecode import CTCBeamDecoder

def ctc_remove_successives_identical_ind(ind):
    res = ""
    for i in range(len(ind)):
        if len(res) > 0 and res[len(res)-1] == ind[i]:
             continue
        res+=(ind[i])

    res=re.sub("(<ctc>)+", '',res).strip(" ")
    return res

def test(model, htr_dataset, test_loader, device, bs=20, lmFile=None, beam=200, gsf=1.0, wip=0.0, verbosity=True):
    
    CTC_BLANK = 1
    
    if (verbosity):
        charVoc=htr_dataset.get_charVoc();
        print("models= %s %i models"%(htr_dataset.get_charVoc(),htr_dataset.get_num_classes()))
        print("CTC_BLANK= %s"%(charVoc[CTC_BLANK]))
        print('Space symbol= \"%s\"'%(htr_dataset.get_spaceChar()))
    
    model.eval()    
    
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
        

        ref, hyp = list(), list()
        
        if (verbosity and not lmFile):
            tq=tqdm(test_loader,colour='cyan', desc='Test', disable=True)
        else:
            tq=tqdm(test_loader,colour='cyan', desc='Test')

        for ((x, input_lengths),(y,target_lengths), bIdxs) in tq:
            x = x.to(device)
            #save_image(x, f"out.png", nrow=1); break

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
            
            assert len(output) == len(target_lengths)
           
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
                        
            

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cer = fastwer.score(hyp, ref, char_level=True)
    wer = fastwer.score(hyp, ref, char_level=False)

   
    print('\n'+'#'*30, file=sys.stderr)
    print(f'# Model\'s params: {total_params}', file=sys.stderr)
    print("             CER: {:5.2f}%".format(cer), file=sys.stderr)
    print("             WER: {:5.2f}%".format(wer), file=sys.stderr)
    print('#'*30, file=sys.stderr)

    return cer, wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing a HTR model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lm-model', type=str, help='laguage model file (Arpa/Kenlm format)', default=None)
    parser.add_argument('--beam-width', type=int, help='this sets up how broad the beam search is', default=200)
    parser.add_argument('--gsf', type=float, help='this sets up Grammar scale factor', default=1.0)
    parser.add_argument('--wip', type=float, help='this sets up Word insertion penalty', default=0.0)
    parser.add_argument('--batch_size', type=int, help='image batch-size', default=24)
    parser.add_argument('--gpu', type=int, default=None, help='used gpu')
    parser.add_argument("--verbosity", action="store_true",  help="increase output verbosity",default=False) 
    parser.add_argument('model', type=str, help='PyTorch model file')
    parser.add_argument('dataset', type=str, help='dataset path')

    args = parser.parse_args()
    print ("\n"+str(sys.argv) )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu is not None:
        if args.gpu > torch.cuda.device_count() or args.gpu < 0:
            sys.exit("ERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()-1))
        torch.cuda.set_device(args.gpu)
    else:
        device = auto_cuda()
        torch.cuda.set_device(device)
        
    print("selected GPU %i"%(torch.cuda.current_device()))
          
    # Load model in memory
    state = torch.load(args.model, map_location=device)
    model = state['model']
   
    # Get the sequence of transformations to apply to images 
    img_transforms = procImg.get_tranform(state['line_height'])
          
    htr_dataset = HTRDataset(args.dataset,
                            state['spaceSymbol'],
                            transform=img_transforms,
                            charVoc=state['codec'])

    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    test_loader = torch.utils.data.DataLoader(htr_dataset,
                                            batch_size = args.batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = False,
                                            collate_fn = ctc_collate)

    test(model, htr_dataset, test_loader, device,
         bs=args.batch_size,
         lmFile=args.lm_model,
         beam=args.beam_width,
         gsf=args.gsf,
         wip=args.wip,
         verbosity=args.verbosity)
  
    
    sys.exit(os.EX_OK)
