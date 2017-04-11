
-- local pl = (require 'pl.import_into')()

require 'torch'
require 'image'
local li_dataset = require 'li_dataset'

torch.setdefaulttensortype('torch.FloatTensor')





local opt = lapp [[
Liveness detection dataset script
This file can be used to divide training and test data from original data
Main options
  -n          (default 264)           Use only N samples for training Alive
  -m          (default 88)			  Use only N samples for training the Others
  -s            (default 0)           Select 0s and 2s
  -v, --nonval                        Make Validation set from half of the test set
]]

-- catch option errors
-- assert((opt.s !=0 or opt.s !=2))
assert((opt.s == 0) or (opt.s == 2) , "option -s should be 0 or 2")

local train
local test
local valid
train, test, valid = li_dataset.makeDataset(opt.n, opt.m, opt.s, opt.nonval)
train = li_dataset.shuffleData(train)
test = li_dataset.shuffleData(test)

print("train size : ", train.data:size())
print("valid size : ", valid.data:size())
print("test size : ", test.data:size())
if valid then
	valid = li_dataset.shuffleData(valid)
	-- print ("I'm valind in")
end

--[[ Setting Output Directory and Save files
	Configure your output file name at below codes
]]
local rootPath = '/home/park/DBs/FAKE/livedet2009/rePark'
local identix = '/Identix/'
local outputPath = rootPath .. identix
local trName = "train"..opt.s.."s".."Data.t7"
local teName = "test"..opt.s.."s".."Data.t7"
local vaName = "valid"..opt.s.."s".."Data.t7"

torch.save(outputPath..trName,train)
torch.save(outputPath..teName,test)

if valid then
	torch.save(outputPath..vaName,valid)
end
