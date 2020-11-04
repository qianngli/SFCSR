import torch
import torch.nn as nn
import math  
       
    
class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(TwoCNN, self).__init__()

        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1)))
               
    def forward(self, x):
    
        out = self.body(x)
        out = torch.add(out, x)
        
        return out             

class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))            

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)

    def forward(self, x): 
        out = x
        for i in range(2):
              
            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)
    
        out = torch.add(out, x)        
        return out
                                                                                                                                                                                                            
class SFCSR(nn.Module):
    def __init__(self, args):
        super(SFCSR, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
 
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
                                                        
        ThreeHead = []
        ThreeHead.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        ThreeHead.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))  
        self.ThreeHead = nn.Sequential(*ThreeHead)
        

        TwoHead = []
        TwoHead.append(wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        self.TwoHead = nn.Sequential(*TwoHead)

        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1))))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1))))
            TwoTail.append(nn.PixelShuffle(3))  

        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1))))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)
                        	 
        twoCNN = []
        for _ in range(self.n_module):
            twoCNN.append(TwoCNN(wn, n_feats))
        self.twoCNN = nn.Sequential(*twoCNN)
        
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))
        
        threeCNN = []
        for _ in range(self.n_module):
            threeCNN.append(ThreeCNN(wn, n_feats))
        self.threeCNN = nn.Sequential(*threeCNN)
      
        reduceD = []
        for _ in range(self.n_module):
            reduceD.append(wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1)) )
        self.reduceD = nn.Sequential(*reduceD)
                                  
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1))   
        
        threefusion = []               
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))          
        self.threefusion = nn.Sequential(*threefusion)
        

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1)) 
        
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1))    
        
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         
        
        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y
        
        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
        
