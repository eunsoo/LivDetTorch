local pl = (require 'pl.import_into')()

require 'torch'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local dataset = {}

-- Private function declaration
local merge_dataset
local prune_dataset

dataset.trAlive  = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Train/cAlivePatchesTrain.bin"
dataset.trGelatin = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Train/cGelatinPatchesTrain.bin"
dataset.trPlaydoh = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Train/cPlaydohPatchesTrain.bin"
dataset.trSilicone = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Train/cSiliconePatchesTrain.bin"

dataset.vaAlive  = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Test_Val/Val/cAlivePatchesVal.bin"
dataset.vaGelatin ="/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Test_Val/Val/cGelatinPatchesVal.bin"
dataset.vaPlaydoh ="/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Test_Val/Val/cPlaydohPatchesVal.bin"
dataset.vaSilicone = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Test_Val/Val/cSiliconePatchesVal.bin"

dataset.trPatches = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/trPatchesData.t7"
dataset.vaPatches = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/vaPatchesData.t7"

-------------------------------------------------
-- Main Interface
-------------------------------------------------

-- Returns the train dataset
-- nbr_examples is optional and allows to get only a subset of the training samples
-- use validation allows to return both a test and validation set (split is done
-- the same way as the paper from Sermanet adnd LeCun)
-- Warning: if the number of examples is not limited, the dataset is ordered by class
-- If the number of examples is limited, the subset will be shuffled.
function dataset.get_train_dataset(nbr_examples, use_validation)
  local train_dataset = torch.load(dataset.trPatches)
  local validation_dataset = torch.load(dataset.vaPatches)

  if not use_validation then
    -- Merge both train and validation set to have the full train set
    local full_train_dataset = {}
    -- Create the full dataset with the proper size
    local data_size = train_dataset.data:size()
    local label_size = train_dataset.label:size()
    local nbr_train_examples = train_dataset.data:size(1)
    local nbr_val_examples = validation_dataset.data:size(1)
    data_size[1] = nbr_train_examples + nbr_val_examples
    label_size[1] = nbr_train_examples + nbr_val_examples
    full_train_dataset.data = torch.Tensor(data_size)
    full_train_dataset.label = torch.Tensor(label_size)
    -- Copy the data in the full dataset
    full_train_dataset.data:narrow(1,
                                  1,nbr_train_examples):copy(
                                  train_dataset.data)
    full_train_dataset.data:narrow(1,
                                  nbr_train_examples+1,
                                  nbr_val_examples):copy(
                                  validation_dataset.data)
    full_train_dataset.label:narrow(1,
                                  1,nbr_train_examples):copy(
                                  train_dataset.label)
    full_train_dataset.label:narrow(1,
                                  nbr_train_examples+1,
                                  nbr_val_examples):copy(
                                  validation_dataset.label)
    validation_dataset = nil
    train_dataset = full_train_dataset
  end


  -- Limit the number of samples if required by the user
  if nbr_examples and nbr_examples ~= -1 then
    train_dataset = prune_dataset(train_dataset, nbr_examples)
  end

  return train_dataset, validation_dataset
end

----------------------------------------------------
-- Test data loader
----------------------------------------------------
-- Returns the test dataset
-- nbr_examples is optional and allows to get only a subset of the testing samples
-- Warning: if the number of examples is not limited, the dataset is ordered by class
-- If the number of examples is limited, the subset will be shuffled.
function dataset.get_test_dataset(nbr_examples)
  local test_dataset = torch.load(dataset.vaPatches)

  -- Limit the number of samples if required by the user
  if nbr_examples and nbr_examples ~= -1 then
    test_dataset = prune_dataset(test_dataset, nbr_examples)
  end
  return test_dataset
end
-------------------------------------------------
-- Make dataset from each saved bin files
-------------------------------------------------
function dataset.makeDataset()
  -- dataset.download_generate_bin()
  -- print("Please ",nAlive)
  local tr_alive
  local tr_gelatin
  local tr_playdoh
  local tr_silicone

  local va_alive
  local va_gelatin
  local va_playdoh
  local va_silicone


  tr_alive = torch.load(dataset.trAlive)
  tr_gelatin = torch.load(dataset.trGelatin)
  tr_trplaydoh = torch.load(dataset.trPlaydoh)
  tr_silicone = torch.load(dataset.trSilicone)

  va_alive = torch.load(dataset.vaAlive)
  va_gelatin = torch.load(dataset.vaGelatin)
  va_playdoh = torch.load(dataset.vaPlaydoh)
  va_silicone = torch.load(dataset.vaSilicone)

  print("Training and Test data are loaded")
  
  local train_dataset
  local valid_dataset

  train_dataset = merge_dataset(tr_alive, tr_gelatin, tr_trplaydoh, tr_silicone)
  valid_dataset = merge_dataset(va_alive, va_gelatin, va_playdoh, va_silicone)
    
  print("Validation data are loaded")
  return train_dataset, valid_dataset
end

--[[ Data shuffling function]]
function dataset.shuffleData(dataset)
  local shuffled={}
  local nData = dataset.data:size(1)
  local randperm = torch.randperm(nData)
  shuffled.data = torch.Tensor(dataset.data:size())
  shuffled.label = torch.Tensor(dataset.label:size())

  for i=1,nData do
    shuffled.data[i]:copy(dataset.data[randperm[i]])
    shuffled.label[i]:copy(dataset.label[randperm[i]])
  end
  collectgarbage()
  return shuffled
end

function dataset.labelChange(dataset)
  local nData = dataset.label:size(1)
--  print(dataset.label:size())
  for i=1, nData do
    if(dataset.label[{i,1}] > 2) then
      dataset.label[i] = 2
    end
  end 
  return dataset
end



-- Normalize the given dataset
-- You can specify the mean and std values, otherwise, they are computed on the given dataset
-- Return the mean and std values
function dataset.normalize_global(dataset, mean, std)
  local std = std or dataset.data:std()
  local mean = mean or dataset.data:mean()
  dataset.data:add(-mean)
  dataset.data:div(std)
  return mean, std
end

-- Locally normalize the dataset
function dataset.normalize_local(dataset)
  require 'image'
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(1,norm_kernel)
  local batch = 50 -- Can be reduced if you had memory issues
  local dataset_size = dataset.data:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset.data:narrow(1,i,local_batch))
    dataset.data:narrow(1,i,local_batch):copy(normalized_images)
  end
end

-------------------------------------------------
-- Private function
-------------------------------------------------

prune_dataset = function(dataset, nbr_examples)
  -- Limit the number of samples if required by the user
  assert(nbr_examples and nbr_examples > 1 and nbr_examples < dataset.data:size(1),
         'Invalid number of examples required, not within dataset range.')

  local randperm = torch.randperm(dataset.data:size(1))
  local subset_data = torch.Tensor(nbr_examples, 1, 96, 96)
  local subset_label = torch.Tensor(nbr_examples, 1)
  for i=1,nbr_examples do
    subset_data[i]:copy(dataset.data[randperm[i]])
    subset_label[i]:copy(dataset.label[randperm[i]])
  end
  dataset.data = subset_data
  dataset.label = subset_label
  collectgarbage()

  return dataset
end

merge_dataset = function(alive, gelatin, playdoh, silicone)
  local full_train_dataset = {}
  -- Create the full dataset with the proper size
  -- Sum up all training data
  local data_size = alive.data:size()
  local label_size = alive.label:size()
  local aNum = alive.data:size(1)
  local gNum = gelatin.data:size(1)
  local pNum = playdoh.data:size(1)
  local sNum = silicone.data:size(1)
  data_size[1] = aNum+gNum+pNum+sNum
  label_size[1] = aNum+gNum+pNum+sNum
  full_train_dataset.data = torch.Tensor(data_size)
  full_train_dataset.label = torch.Tensor(label_size)


  full_train_dataset.data:narrow(1,1,aNum):copy(alive.data)
  full_train_dataset.data:narrow(1,aNum+1,gNum):copy(gelatin.data)
  full_train_dataset.data:narrow(1,aNum+gNum+1,pNum):copy(playdoh.data)
  full_train_dataset.data:narrow(1,aNum+gNum+pNum+1,sNum):copy(silicone.data)

  full_train_dataset.label:narrow(1,1,aNum):copy(alive.label)
  full_train_dataset.label:narrow(1,aNum+1,gNum):copy(gelatin.label)
  full_train_dataset.label:narrow(1,aNum+gNum+1,pNum):copy(playdoh.label)
  full_train_dataset.label:narrow(1,aNum+gNum+pNum+1,sNum):copy(silicone.label)

  return full_train_dataset
end

return dataset

