torch.setdefaulttensortype('torch.FloatTensor')
require 'torch'
require 'image'


function generate_dataset(images_directories, classId, channel, imgWidth, imgHeight)
--  assert(images_directories, "A parent path is needed to generate the dataset")

  local main_dataset = {}
  main_dataset.nbr_elements = 0

  table.sort(images_directories)

  for image_index, image_path in ipairs(images_directories) do
    local image_data = image.load(image_path)

    main_dataset.nbr_elements = main_dataset.nbr_elements + 1
    local label = torch.Tensor{classId}
    main_dataset[main_dataset.nbr_elements] = {image_data, label}

    if image_index % 100 == 0 then
      collectgarbage()
    end
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

[[--
 This Part is for making torch.save file from testing data
--]]
-- These paths should not be changed
local rootPath = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Test/Patches/"
-- local identix = '/Identix'
-- local biometrika = '/Biometrika'
-- local crossmatch = '/CrossMatch'
local alive = '/Alive'
local gelatin = '/Gelatin'
local playdoh = '/PlayDoh'
local silcone = '/Silicone'
-- local zeroS = '/0s'
local twoS = '/2s'


local outputPath = rootPath .. '/' -- specify output folder


for file in paths.files(fileFolder) do
  if file:find(ext .. '$') then
    table.insert(imgFiles, paths.concat(fileFolder,file))
  end
end

[[--
 This Part is for making torch.save file so if you want to use this function, Comment out~
--]]

-- -- These paths should not be changed
-- local rootPath = "/home/park/mnt/DBs/FAKE/rePark/Version_0_0/Identix/Train/SegRot/Patches"
-- -- local identix = '/Identix'
-- -- local biometrika = '/Biometrika'
-- -- local crossmatch = '/CrossMatch'
-- local alive = '/Alive'
-- local gelatin = '/Gelatin'
-- local playdoh = '/PlayDoh'
-- local silcone = '/Silicone'
-- -- local zeroS = '/0s'
-- local twoS = '/2s'

-- -- local sensorFolder = rootPath .. identix
-- local aliveImgF =  rootPath .. alive .. twoS
-- local gelatinImgF = rootPath .. gelatin .. twoS
-- local pladohImgF = rootPath .. playdoh .. twoS
-- local silconeImgF = rootPath .. silcone .. twoS

-- -- Configure part -- 

-- ----- Configure Part
-- local outputPath = rootPath .. '/' -- specify output folder

-- ------------------------------------------------------------
-- local zeroData = "silconePatchesTrain.bin" -- specify output name
-- ------------------------------------------------------------

-- local ext = 'png' -- specify file extension
-- local channel = 1 -- specify image channel
-- local imgWidth = 96 -- specify image width
-- local imgHeight = 96 -- specify image height

-- ------------------------------------------------------------
-- local classId = 4 -- specify class ID
-- ------------------------------------------------------------

-- ------------------------------------------------------------
-- local fileFolder = silconeImgF -- specify folder
-- ------------------------------------------------------------

-- local imgFiles = {}


-- for file in paths.files(fileFolder) do
--   if file:find(ext .. '$') then
--     table.insert(imgFiles, paths.concat(fileFolder,file))
--   end
-- end

-- local dataset = generate_dataset(imgFiles,classId,channel,imgWidth,imgHeight)
-- torch.save(outputPath..zeroData, dataset)

print "Thank you for using"