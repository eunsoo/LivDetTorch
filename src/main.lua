require 'torch'
require 'nn'

local livdec = require 'livdec'
local lapp = require 'pl.lapp'

torch.setdefaulttensortype('torch.FloatTensor')

-- Delete me -- 
--paths.dofile("realImage_test.lua")
--------- Delete me end ----------


local opt = lapp [[
GTSRB Training script
Main options
  -o,--output   (default "")              Output model
  --eval                                  Only run eval
  --val                                   Use a validation set instead of the test set
  --script                                Write accuracy on last line of output for benchmarks
  --no_cuda                               Do not use CUDA
  -n            (default 20000)           Use only N samples for training
  -e            (default 10)              Number of epochs
  --no_bi          

Network generation
  --cnn         (default "108,200,100")   Basic network factory parameters (see doc for more details)
  --net         (default "")              Path to user defined module for networks
  --ms                                    Use multi-scale network

Normalization
  --no_norm                               Do not globally normalize the training and test samples
  --no_lnorm                              Do not locally normalize the training and test samples
  --no_cnorm                              Do not use contrastive normalization in conv

Learning hyperparameters
  -s,--seed     (default 1)               Random seed value (-1 to disable)
  -b,--bs       (default 20)              Mini batch size
  --lr          (default 0.01)            Initial learning rate
  --lrd         (default 0)               Learning rate decay
  --wd          (default 0)               Weight decay
  --mom         (default 0)               Momentum

Spatial transformer options
  --st                                    Add a spatial transformer module
  --locnet      (default "30,60,30")      Localization network parameters
  --locnet2     (default "")              Localization network parameters for second st
  --locnet3     (default "")              Localization network parameters for third st (idsia net only)
  --rot                                   Force the st to use rotation
  --sca                                   Force the st to use scale
  --tra                                   Force the st to use translation
]]


if opt.seed > 0 then
  torch.manualSeed(opt.seed)
  math.randomseed(opt.seed)
end

if not opt.no_cuda then
  require 'cunn'
  print("--cunn is loaded")
end

print("--Loading training data...")
local train_dataset, test_dataset
if opt.val then
  -- In this case our variable 'test_dataset' contains the validation set
  train_dataset, test_dataset = livdec.dataset.get_train_dataset(opt.n, true)
else
  -- Actually 
  train_dataset = livdec.dataset.get_train_dataset(opt.n)
  test_dataset = livdec.dataset.get_test_dataset()
end

if not opt.no_bi then
  train_dataset = livdec.dataset.labelChange(train_dataset)
  test_dataset = livdec.dataset.labelChange(test_dataset)
  print("Alive vs. Fake")
  else
  print ("Alive vs. Gelatin vs. Playdoh vs. Silicone")
end

print("--Load finish.")
print(train_dataset.data:size())
print(test_dataset.data:size())


local mean, std
if not opt.no_norm then
  print('Performing global normalization...')
  mean, std = livdec.dataset.normalize_global(train_dataset)
  livdec.dataset.normalize_global(test_dataset, mean, std)
end
if not opt.no_lnorm then
  print('Performing local normalization...')
  livdec.dataset.normalize_local(train_dataset)
  livdec.dataset.normalize_local(test_dataset)
end

local network
if opt.eval then
  if opt.output then
    print('Loading network from '..opt.output)
    network = torch.load(opt.output)
  else
    error('Must supply the network to use in eval mode')
  end
else
  print('Building the network...')
  network = livdec.networks.new(opt)
end
local criterion = nn.CrossEntropyCriterion()
if not opt.no_cuda then
  network = network:cuda()
  criterion = criterion:cuda()
end
print(network)

print("Initializing the trainer...")
livdec.trainer.initialize(network, criterion, opt)

local _, accuracy

if opt.eval then
  _, accuracy = livdec.trainer.test(test_dataset)
else
  local epoch = 1
  while opt.e == -1 or opt.e >= epoch do
    print('Starting epoch '..epoch)

    livdec.trainer.train(train_dataset)

    _, accuracy = livdec.trainer.test(test_dataset)

    if opt.output ~= '' then
      torch.save(opt.output, network)
      torch.save(opt.output.."norm", {mean, std})
    end
    epoch = epoch + 1
  end
end

if opt.script then
  print(accuracy)
end

print("----- Finshed ------")








