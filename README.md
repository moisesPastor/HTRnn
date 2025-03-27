## Handwritten Text Recognition with PyTorch
This handwritten text recognition model implements CNN+LSTM with CTC loss and beam search CTC decoding. The ctcdecode library must be installed along with other required dependencies.

If multiple GPUs are available, you can specify which one to use with --gpu NumGPU. If not specified, the program selects the GPU with the lowest memory usage.

During testing, if no language model is specified, the program does not use CTCBeamDecoder, improving response time by more than tenfold.
