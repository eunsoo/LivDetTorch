
-- local pl = (require 'pl.import_into')()

require 'torch'
require 'image'
local li_dataset = require 'li_dataset_new'

torch.setdefaulttensortype('torch.FloatTensor')

-- local opt = lapp [[
-- Liveness detection dataset script
-- This file can be used to divide training and test data from original data
-- Main options
--   -n          (default 264)           Use only N samples for training Alive
--   -m          (default 88)			  Use only N samples for training the Others
--   -s            (default 0)           Select 0s and 2s
--   -v, --nonval                        Make Validation set from half of the test set
-- ]]

-- catch option errors
-- assert((opt.s !=0 or opt.s !=2))

local train
local valid
train, valid = li_dataset.makeDataset()
train = li_dataset.shuffleData(train)
valid = li_dataset.shuffleData(valid)

print("train size : ", train.data:size())
print("valid size : ", valid.data:size())

--[[ Setting Output Directory and Save files
	Configure your output file name at below codes
]]
local outputPath = '/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Clean/'
local trName = "trPatchesData.t7"
local vaName = "vaPatchesData.t7"


torch.save(outputPath..trName, train)
torch.save(outputPath..vaName, valid)

