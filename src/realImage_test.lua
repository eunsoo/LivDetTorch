require 'torch'
require 'nn'
require 'cunn'
require 'image'

local lapp = require 'pl.lapp'
local livdec = require 'livdec'


print("--cunn is loaded")
local opt = lapp [[
LivDec Test Options
Main options
  -f (default "Gelatin")    material name
  
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
  
]]

-- Image List table making --
local function makeTable(imgFolder, ext, nSample)
  local direc = paths.dir(imgFolder)
  table.remove(direc,1)
  table.remove(direc,1)
  local realFile = {}
--  local ext = 'png'
  for i, v in ipairs(direc) do
      realFile[v] ={}
      for file in paths.files(imgFolder..'/'..v) do
          if file:find(ext .. '$') then
              table.insert(realFile[v], paths.concat(imgFolder..'/'..v,file))
          end
      end
  end
  local sImageList ={}
--  local nSample = 10
  for f in pairs(realFile) do
    local randperm = torch.randperm(#realFile[f])
    local tempList = {}
    for i=1, nSample do
    tempList[i] = realFile[f][randperm[i]]
    end
    sImageList[f] = tempList
  end
  return sImageList
 end

local function generate_dataset(images_directories, classId, channel, imgWidth, imgHeight)
--  assert(images_directories, "A parent path is needed to generate the dataset")

  local main_dataset = {}
  main_dataset.nbr_elements = 0

  table.sort(images_directories)

  for image_index, image_path in ipairs(images_directories) do
    local image_data = image.load(image_path)

    main_dataset.nbr_elements = main_dataset.nbr_elements + 1
    local label = torch.Tensor{classId}
    main_dataset[main_dataset.nbr_elements] = {image_data, label}
  end

  -- Store everything as proper torch Tensor now that we know the total size
  local main_data = torch.Tensor(main_dataset.nbr_elements, channel, imgWidth, imgHeight)
  local main_label = torch.Tensor(main_dataset.nbr_elements, 1)
  for i,pair in ipairs(main_dataset) do
    main_data[i]:copy(main_dataset[i][1])
    main_label[i]:copy(main_dataset[i][2])
  end
  main_dataset = {}
  main_dataset.data = main_data
  main_dataset.label = main_label
  return main_dataset
end


-- Load Network files
local preTrain = "/home/park/Dropbox/100.Projects/LuaWorkspace/LiveDetPatches/src/idsia_net.t7"
local preNorm = "/home/park/Dropbox/100.Projects/LuaWorkspace/LiveDetPatches/src/idsia_net.t7norm"

-- Test Folder
local testFolder = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/Test_Val/Test"
local two ='2s'
local classId 
print("--Materials : " .. opt.f)

if(opt.f=="Alive") then
  classId = 1
  else
  classId = 2
end
print("--Materials : " .. opt.f)
print("ClassId : " .. classId)



-- Folder selection
local imgFolder = testFolder .. "/" .. opt.f .. "/" ..two
--print(imgFolder)
local ext ="png"
local nSample = 11
local imageList = makeTable(imgFolder, ext, nSample)

-- Network Loader --
local network, criterion

print("Loading network from "..preTrain)
network = torch.load(preTrain)
local criterion = nn.CrossEntropyCriterion()
network = network:cuda()
criterion = criterion:cuda()

--images_directories, classId, channel, imgWidth, imgHeight
livdec.trainer.initialize(network,criterion,opt)
-- Testing -- 

local _
local accuracy = {}
for k in pairs(imageList) do
  
  local testData = generate_dataset(imageList[k], classId, 1, 96, 96)
  if not opt.no_norm then
    print('--Performing global normalization...')
    livdec.dataset.normalize_global(testData, preNorm[1], preNorm[2])
  end
  
  if not opt.no_lnorm then
    print('--Performing local normalization...')
    livdec.dataset.normalize_local(testData)
  end
  _, accuracy[k] = livdec.trainer.test_real(testData)
--  print(accuracy)
--  print(testData.data:size())
--  print(testData.label:size())
end

for acc in pairs(accuracy) do
  print(acc .. " : " .. accuracy[acc])
end
---- -------------------------------------------------------
-- error check
local denom = 0
local norm = 0
for v in pairs(accuracy) do
  denom = denom+1
  if(accuracy[v] < 0.5) then
    norm = norm+1
  end
end
print(denom, norm)
print("Accuracy of ".. opt.f .. " : " .. norm/denom)


--print(network)



 