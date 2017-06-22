-- This script is a easy function set containing some convinient functions by me.
-- Just for nngraph model
require 'nn'
--require 'torch'
function printNet(net)

    for i = 1, net:size(1) do
        print(string.format("%d: %s", i, net.modules[i]))
    end
    
end

function reset_Globals()
    conv_enc1_layer = nil
    maxcoor1_layer = nil
    conv_enc1_layer = nil
    centers_lt1_layer = nil
    centers_rb1_layer = nil

    conv_enc2_layer = nil
    maxcoor2_layer = nil
    conv_enc2_layer = nil
    centers_lt2_layer = nil
    centers_rb2_layer = nil

    conv_enc3_layer = nil
    maxcoor3_layer = nil
    conv_enc3_layer = nil
    centers_lt3_layer = nil
    centers_rb3_layer = nil

    conv_enc4_layer = nil
    maxcoor4_layer = nil
    conv_enc4_layer = nil
    centers_lt4_layer = nil
    centers_rb4_layer = nil
    collectgarbage()
end



function findModule(net, layer)
    assert(type(layer) == 'string', 'layer name must be string')
    for i = 1, net:size(1) do
        --print(tostring((net.modules[i])))
        if tostring(net.modules[i]) == layer then
            return i
        end
    end
    return -1
end

-- input should be 3 dimensions or 4 dimensions.
-- the net should be the whole net!
-- Then get the intermediate layer which need to check gradients.
-- layer should be a number indicating which layer to get result.
-- Eg. If just add two swaps, and swaps are net.module[40] and net.module[45]
-- We need to pass layer 41 and 46 as we want to get output after swapX.
function check_gradient(net, layer, input, epsilon)
    local N, C, H, W
    if input:nDimension() == 3 then
        C, H, W = input:size()
        N = 1
        input = input:view(1,C,H,W)
    elseif input:nDimension() == 4 then
        N, C, H, W = input:size()
    else
        error('input dimension error!')
    end

    local net = net:clone()
    local out = net:forward(input)
    net:forward(torch.add(input, epsilon))
    local indexNode, swap_out1, swap_out2
    for indexNode, node in pairs(net.forwardnodes) do
        if indexNode == layer then
            if node.data.module then
                swap_out1= node.data.module.output:clone()
            end
            
        end
    end
    net:forward(torch.add(input, -epsilon))

    for indexNode, node in pairs(net.forwardnodes) do
        if indexNode == layer then
            if node.data.module then
                swap_out2 = node.data.module.output:clone()
            end
            
        end
    end

    local delta = swap_out1 - swap_out2
    local grad = torch.div(delta, 2*epsilon)
    print('checking '..layer..' gradient sum...'..torch.sum(grad))

end

function cal_feat_mask(inMask, conv_layers, threshold)

    assert(inMask:nDimension() == 2,'mask must be 2 dimensions')
    local lnet = nn.Sequential()
    for id_net = 1, conv_layers do
        local conv = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
        conv.weight:fill(1/16)
        lnet = lnet:add(conv)
    end

    lnet = lnet:cuda()
    local output = lnet:forward(inMask:view(1,inMask:size(1),inMask:size(2)))
    assert(inMask:size(1)/(torch.pow(2,conv_layers)) == output:size(2))
    output = output:squeeze()  -- remove batch_size channel
    local fLen = output:size(1)
    output = output:float()

    local i,j
    for i = 1, fLen do
        for j = 1, fLen do
            local v = torch.mean(output[{{i},{j}}])
            output[{{i},{j}}] = v > threshold and 1 or 0
        end
    end

    return output
end

-- mask_global should be 1*256*256
function create_gMask(pattern,mask_global, MAX_SIZE, opt)
   local mask, wastedIter
   wastedIter = 0
	if opt.test == 1 then
			local mask_temp = image.load('mask.png',1,'byte')
	--image.save('1.png',mask_temp)
			mask_temp = mask_temp/255
			mask = mask_temp
			--mask = mask:squeeze()
			--print(mask:size())
	else
	   while true do
		 local x = torch.uniform(1, MAX_SIZE-opt.fineSize)
		 local y = torch.uniform(1, MAX_SIZE-opt.fineSize)
		 mask = pattern[{{y,y+opt.fineSize-1},{x,x+opt.fineSize-1}}]  -- view, no allocation
		 local area = mask:sum()*100./(opt.fineSize*opt.fineSize)
		 if area>20 and area<30 then  -- want it to be approx 75% 0s and 25% 1s
		    -- print('wasted tries: ',wastedIter)
		    break
		 end
		 wastedIter = wastedIter + 1
	   end
	end
		--[[local x = 65
		local y = 65--torch.uniform(1, 128)
		local mask = torch.zeros(opt.fineSize , opt.fineSize)
		mask[{{x,x+127},{y,y+127}}]=1]]

   torch.repeatTensor(mask_global,mask,opt.batchSize,1,1)
   return mask_global--,x,y
end




--************* Desperate method *************************
-- -- only for u-net
-- -- default 4*4 with stride = 2 and pad = 1
-- -- inMask should be binnary with value in mask region equaling 1.
-- -- Mention: inMask should be size of 2^n. 
-- -- conv_layers: number of such convs.
-- function cal_feat_mask(inMask, conv_layers)

--     assert(inMask:nDimension() == 2,'mask must be 2 dimensions')
--     local stride = 2
--     local kernelSize = 4
--     local pad = 1
--     local maxConv = 8 -- assuming that the input is 256*256
--     -- one more conv makes feature map decrease 2 times.
--     local finalF = torch.pow(2, maxConv - conv_layers) -- how many super-pixels in a line in featuremap
--     local pointSizeF = torch.floor(inMask:size(2)/finalF)
--     local fMask = torch.Tensor(finalF, finalF):zero():cuda()
--     -- maybe need to optimize
--     local i,j
--     for i = 1, finalF do
--         for j = 1, finalF do
--             local lt_x =  pointSizeF*(i - 1) + 1
--             local rb_x =  pointSizeF*(i - 1) + 1 + pointSizeF - 1
--             local lt_y =  pointSizeF*(j - 1) + 1
--             local rb_y =  pointSizeF*(j - 1) + 1 + pointSizeF - 1
--             local meanV = torch.mean(inMask[{{lt_x, rb_x},{lt_y, rb_y}}])
--             fMask[{{i},{j}}] = meanV > 0.5 and 1 or 0
--         end
--     end
--  --   print('elapsed_time '..tm1:time().real)

--     return fMask
-- end



-- try to make nngraph normal net
-- function defineG_unet_swap_normal(input_nc,output_nc,ngf)
--   local net_G=nil
--     netG = nn.Sequential()
--     netG:add(nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1))
--     -- local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)

--     -- input is (ngf) x 128 x 128
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 2))
--     -- -- input is (ngf) x 128 x 128
--     -- local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2)

--     -- e3 input is (ngf * 2) x 64 x 64
--     netG:add(nn.LeakyReLU(0.2, true):add(nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 4))
--     -- -- input is (ngf * 2) x 64 x 64
--     -- local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 4) 

--     -- e4 input is (ngf * 4) x 32 x 32
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 8))
--     -- input is (ngf * 4) x 32 x 32
--     -- local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)

--     -- input is (ngf * 8) x 16 x 16
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 8))
--     -- input is (ngf * 8) x 16 x 16
--     -- local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)

--     -- input is (ngf * 8) x 8 x 8
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 8))    
--     -- input is (ngf * 8) x 8 x 8
--     -- local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)

--     -- input is (ngf * 8) x 4 x 4
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 8))
--     -- input is (ngf * 8) x 4 x 4
--     -- local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)

--     -- input is (ngf * 8) x 2 x 2
--     netG:add(nn.LeakyReLU(0.2, true)):add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)):add(nn.InstanceNormalization(ngf * 8))
--     -- input is (ngf * 8) x 2 x 2
--     -- local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)

--     -- input is (ngf * 8) x 1 x 1
--     net:add()
--     local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
--     -- input is (ngf * 8) x 2 x 2
--     local d1 = {d1_,e7} - nn.JoinTable(2)
--     local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
--     -- input is (ngf * 8) x 4 x 4
--     local d2 = {d2_,e6} - nn.JoinTable(2)
--     local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
--     -- input is (ngf * 8) x 8 x 8
--     local d3 = {d3_,e5} - nn.JoinTable(2)
--     local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
--     -- input is (ngf * 8) x 16 x 16
--     local d4 = {d4_,e4} - nn.JoinTable(2)
--     local d5_ = d4 - nn.ReLU(true) - swap1 - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 4)
--     -- input is (ngf * 4) x 32 x 32
--     local d5 = {d5_,e3} - nn.JoinTable(2)
--     local d6_ = d5 - nn.ReLU(true) - swap2 - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2)
--     -- input is (ngf * 2) x 64 x 64
--     local d6 = {d6_,e2} - nn.JoinTable(2)
--     local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf)

--     -- input is (ngf) x128 x 128
--     local d7 = {d7_,e1} - nn.JoinTable(2)
--     local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
--     -- input is (nc) x 256 x 256
--     local o1 = d8 - nn.Tanh()     


-- -- idx means the index counted backwards

--     netG = nn.gModule({e1},{o1})
-- -- print netG 
--     print('netG...')
--    printNet(netG)

--     return netG
-- end



-- padding should equals to kernel_size if padding equals zero
-- and you want to get the same size as the input.
--[[
-- for example:
local input = torch.Tensor(2,8,9)
local padding = 3
print(getAverageTemplate(input, padding))

(1,.,.) =
  1  2  3  3  3  3  3  2  1
  2  4  6  6  6  6  6  4  2
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  2  4  6  6  6  6  6  4  2
  1  2  3  3  3  3  3  2  1

(2,.,.) =
  1  2  3  3  3  3  3  2  1
  2  4  6  6  6  6  6  4  2
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  3  6  9  9  9  9  9  6  3
  2  4  6  6  6  6  6  4  2
  1  2  3  3  3  3  3  2  1
[torch.DoubleTensor of size 2x8x9]
]]
function getAverageTemplate(input, padding)
  local function refleat2DSquareTensor(tensor, dim)
    local t = tensor:clone()
    local flipIdx = math.ceil(t:size(1)/2)
    if dim == 1 then
      for i = 1, flipIdx -1 do
        local tmp = t[{ {i},{} }]:clone()
        t[{ {i},{} }] = t[{ {-i},{} }]:clone()
        t[{ {-i},{} }] = tmp:clone()
      end
    else
        for i = 1, flipIdx -1 do
        local tmp = t[{ {},{i} }]:clone()
        t[{ {},{i} }] = t[{ {},{-i} }]:clone()
        t[{ {},{-i} }] = tmp:clone()
      end
    end
    return t
  end
  local sz = input:size()
  local c = sz[1]
  local h = sz[2]
  local w = sz[3]
  local output = torch.zeros(h, w):typeAs(input)
  local final = padding*padding
  -- First, construct letTen.
  local i = 0
  local letTen = torch.Tensor(padding):apply(function()i = i+1 return i end):repeatTensor(padding,1)
  for i = 2, padding do
    letTen[i] = letTen[1] * i
  end
  -- Second, fill the center with padding*padding.
  output[{{padding+1, -padding-1},{padding+1, -padding-1} }]:fill(final)

  -- Third, fill four corners with leTen and its transformations.
  local flip1letTen = refleat2DSquareTensor(letTen,1)
  local flip2letTen = refleat2DSquareTensor(letTen,2)
  local flip3letTen = refleat2DSquareTensor(flip2letTen,1)
  output[{{1, padding},{1,padding} }] = letTen:clone()  --left_up
  output[{{-padding, -1},{1,padding} }] = flip1letTen:clone() --left_down
  output[{{1, padding},{-padding, -1} }] = flip2letTen:clone()  -- right_up
  output[{{-padding, -1},{-padding,-1} }] = flip3letTen:clone()  --right_down

  --Fourth, fill four white spaces
  local colTen = output[{ {1, padding},{padding} }]:clone()
  local flipcolTen = output[{ {-padding, -1},{padding} }]:clone()
  local rowTen = output[{ {padding},{1, padding} }]:clone()
  local fliprowTen = output[{ {padding},{-padding, -1} }]:clone()
  output[{ {1, padding},{padding+1, -padding-1} }] = colTen:repeatTensor(1, w - 2*padding):clone() --up_center
  output[{ {padding+1, -padding-1},{1, padding} }] = rowTen:repeatTensor(h - 2*padding, 1):clone() --left_center
  output[{ {padding, -padding-1},{-padding, -1} }] = fliprowTen:repeatTensor(h - 2*padding+1, 1):clone() --right_center
  output[{ {-padding, -1},{padding, -padding-1} }] = flipcolTen:repeatTensor(1, w - 2*padding+1):clone() --down_center

  --fifth, expand tensor to c channels
  output = output:repeatTensor(c,1,1)
  return output
end


function Ab()
    assert(1==2)
end
