import torch
noise_dict = {}
size = [(1, 1, 4, 4),(1, 1, 8, 8),(1, 1, 16, 16),(1, 1, 32, 32),(1, 1, 64, 64),(1, 1, 128, 128),(1, 1, 256, 256),(1, 1, 512, 512)]
for s in size: 
    out = torch.rand(s)#.cuda()
    noise = out.new_empty(s).normal_()
    #print(s[2])
    noise_dict[s[2]] = noise
    #print(noise_dict)
