import numpy as np
import matplotlib.pyplot as plt
import torch

class Datagen:
    def __init__(self,dim,mod,means,stddev,num_points):
        self.dim = dim
        self.mod = mod
        self.means = means
        self.stddev = stddev
        self.num_points = num_points
        self.ids = np.arange(1,self.mod*self.num_points + 1,1)
    
    def reshape_means(self):
        self.means = np.array(self.means).reshape(self.mod,self.dim)
    
    def reshape_stddev(self):
        self.stddev = np.array(self.stddev).reshape(self.mod,self.dim)

    def generate_gaussian(self):        
        pointsarr = []
        for row in range(self.mod):
            pointsarr.append(np.random.multivariate_normal(self.means[row],np.diag(self.stddev[row]),self.num_points))
        pointsarr = np.concatenate(pointsarr)
        return pointsarr  
    
    def totensor(self,numpyobj):
        return torch.tensor(numpyobj)

def main():
	
	# with open('input.txt') as f:
	# 	dim = next(f).split()[0]
	# 	print(dim)
	# 	mean = next(f).split()[0]
	# 	print(mean)
	# 	means = [int(x) for x in next(f).split()]
	# 	print(means)
	# 	stddevs = [int(x) for x in next(f).split()]
	# 	print(stddevs)
	# 	points = next(f).split()[0]
	# 	print(points)

	print("Enter dimensionality of data: ")
	dim = int(input())
	
	print("Enter number of modalities/central points: ")
	mod = int(input())

	print("Enter {} number of space seperated mean values".format(mod*dim))
	means = list(map(float,input().split()))

	print("Enter {} number of space seperated standard deviations".format(mod*dim))
	stddev = list(map(float,input().split()))

	print("Enter number of points per modality")
	points = int(input())

	d_obj = Datagen(dim,mod,means,stddev,points)
	d_obj.reshape_means()
	d_obj.reshape_stddev()
	pointsarr = d_obj.totensor(d_obj.generate_gaussian())

	for ind,data in enumerate(pointsarr):
		torch.save(pointsarr[ind], 'data/' + str(ind+1) + '.pt')	

	aa = torch.load('data/200.pt')
	print(aa)

if __name__ == '__main__':
	main()
