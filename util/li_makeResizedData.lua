require 'torch'
require 'image'
dataset = require 'li_dataset'

torch.setdefaulttensortype('torch.FloatTensor')

-- local dataset = {}
local saveFilePath = "/home/park/DBs/FAKE/livedet2009/rePark/Identix/Resized/"

local tr0 = torch.load(dataset.train0s)
local te0 = torch.load(dataset.test0s)
local va0 = torch.load(dataset.valid0s)

local tr2 = torch.load(dataset.train2s)
local te2 = torch.load(dataset.test2s)
local va2 = torch.load(dataset.valid2s)

local sizeList = {512, 256, 128, 64, 48}

for i=1, #sizeList do
	local temp = {}
	temp = dataset.makeResizeDataset(tr0, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."tr0_".. sizeList[i] .. ".t7", temp)
	temp = dataset.makeResizeDataset(te0, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."te0_".. sizeList[i] .. ".t7", temp)
	temp = dataset.makeResizeDataset(va0, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."va0_".. sizeList[i] .. ".t7", temp)

	temp = dataset.makeResizeDataset(tr2, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."tr2_".. sizeList[i] .. ".t7", temp)
	temp = dataset.makeResizeDataset(te2, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."te2_".. sizeList[i] .. ".t7", temp)
	temp = dataset.makeResizeDataset(va2, sizeList[i], sizeList[i])
	torch.save(saveFilePath.."va2_".. sizeList[i] .. ".t7", temp)

	print(sizeList[i], "resizing finished.")
end 

-- for i=1, #sizeList do
-- 	local temp = {}
	
-- end 
-- do  -- make size 512 x 512
	
-- end

-- do  -- make size 256 x 256
-- 	local temp = {}
-- 	temp = dataset.makeResizeDataset(tr0, sizeList[2], sizeList[2])
-- 	torch.save(temp, saveFilePath.."tr0_".. sizeList[2] .. ".t7")
-- 	temp = dataset.makeResizeDataset(te0, sizeList[2], sizeList[2])
-- 	torch.save(te0, saveFilePath.."te0_".. sizeList[2] .. ".t7")
-- 	temp = dataset.makeResizeDataset(va0, sizeList[2], sizeList[2])
-- 	torch.save(va0, saveFilePath.."va0_".. sizeList[2] .. ".t7")
-- end

-- do  -- make size 512 x 512
-- 	local temp = {}
-- 	temp = dataset.makeResizeDataset(tr0, sizeList[1], sizeList[1])
-- 	torch.save(temp, saveFilePath.."tr0_".. sizeList[1] .. ".t7")
-- 	temp = dataset.makeResizeDataset(te0, sizeList[1], sizeList[1])
-- 	torch.save(te0, saveFilePath.."te0_".. sizeList[1] .. ".t7")
-- 	temp = dataset.makeResizeDataset(va0, sizeList[1], sizeList[1])
-- 	torch.save(va0, saveFilePath.."va0_".. sizeList[1] .. ".t7")
-- end





-- image.display(tr0.data[3])
-- print("data size : :", tr0.data:size())
-- print("data label : :", tr0.label:size())

-- print("data label 0: :", tr0.label[3])


