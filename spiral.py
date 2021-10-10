# spiral.py
# ZZEN9444, CSE, UNSW
# Mohammad Reza Hosseinzadeh | z5388543

# import libraries 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# set seed
torch.manual_seed(1234)

### PolarNet
# step1: convert input (x,y) to polar coord (r,a)
# step2: feed (r,a) into FC hidden layer using tanh activation
# step3: output layer with single output using sigmoid activation
class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # FC hidden layer with 2 inputs (x,y)
        self.layer1 = nn.Linear(2, num_hid)
        # FC output layer with 1 output
        self.layer2 = nn.Linear(num_hid, 1)
        # empty object to be filled with activated hidden layer polar coord  
        self.hidden1 = None

    def forward(self, input):
        ### convert cartesian (x,y) to polar coordinates (r,a)
        ## assign radial coordinate (radius), r = sqrt(x**2 + y**2)
        x_coord = input[:, 0]
        y_coord = input[:, 1]
        x_square = torch.square(x_coord)
        y_square = torch.square(y_coord)
        sq_sum = x_square + y_square
        r_polar = torch.sqrt(sq_sum)  
        # flatten r_polar tensor to 1D
        r = r_polar.unsqueeze(1)

        ## assign angular coordinate (azimuth), a = atan2(y,x)
        a_polar = torch.atan2(y_coord, x_coord)
        # flatten a_polar tensor to 1D
        a = a_polar.unsqueeze(1)
        # combine polar coord tensor
        polar = torch.cat((r, a), dim=1)

        ## polar to hidden layer1
        polar_L1 = self.layer1(polar)
        # apply tanh activation to hidden layer1
        self.hidden1 = torch.tanh(polar_L1)

        ## hidden to output layer
        out_sum = self.layer2(self.hidden1)
        # apply sigmoid activation to output layer
        output = torch.sigmoid(out_sum)    

        return output 


### RawNet
# operates on raw input (x,y) without converting to polar
# 2 x FC hidden layers using tanh activation, 
# same number of hidden nodes as determined by parameter num_hid
# 1 x output layer with single output using sigmoid activation
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid, num_hid)
        self.layer3 = nn.Linear(num_hid, 1)

        self.hidden1 = None
        self.hidden2 = None


    def forward(self, input):
        input_L1 = self.layer1(input)
        self.hidden1 = torch.tanh(input_L1)

        hidden1_L2 = self.layer2(self.hidden1)
        self.hidden2 = torch.tanh(hidden1_L2)

        out_sum = self.layer3(self.hidden2)
        output = torch.sigmoid(out_sum)

        return output

def graph_hidden(net, layer, node):
    # hidden function given in the assignment
    xrange = torch.arange(start= -7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start= -6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)),1) 

    # after training, pick the model with best validation accuracy
    with torch.no_grad():   # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        output = net(grid) 
   
        # in hidden layers, the output of tanh() is from -1 to 1. 
        # therefore, value 0, as the middle point, is the threshold
        if layer==1:
            pred = (net.hidden1[:, node]>=0).float()
        else:
            pred = (net.hidden2[:, node]>=0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, 
        pred.cpu().view(yrange.size()[0], xrange.size()[0]), 
        cmap='Wistia') 

