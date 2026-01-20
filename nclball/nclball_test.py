from models.ncl import BallNCL, ball_boundary_uniform, ball_uniform
import torch


dim = 4
lt_dim = int(dim*(dim-1)/2)
mat = torch.zeros((dim, dim))
out = torch.rand((lt_dim))
indexes = torch.triu_indices(dim, dim, offset=1)
triu_indexes = torch.triu_indices(dim, dim, offset=1)
mat = mat.index_put(tuple(triu_indexes), out)
#mat.index_put_(tuple(indexes), out)


mncl = BallNCL(hidden_units=[20, 20, 20],
               lr=1e-3).to('cuda')


x = torch.randn((10, 4)).to(device='cuda')
y = mncl(x)
print(f'Output shape: {y.shape}')

from torch.func import jacrev, vmap

div = vmap(jacrev(mncl.forward_single))
print(f'Single input shape: {x[0].unsqueeze_(0).shape}')
print(f'Single output shape: {mncl.forward(x[0].unsqueeze_(0)).shape}')


print(f'Jacobian shape: {div(x[0].unsqueeze_(0)).shape}')
print(f'Divergence value: {torch.einsum("bii -> b", div(x)[:,:4,:])}')


no_div = vmap(jacrev(mncl.net))
print(x[0].unsqueeze_(0).shape)
print(mncl.forward(x[0].unsqueeze_(0)).shape)
#print(div(x[0].unsqueeze_(0)))

print(no_div(x[0].unsqueeze_(0)).shape)
print(torch.einsum('bii -> b', no_div(x)[:,:4,:]))



from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

pts = ball_boundary_uniform(100, 1, 3)
xi = pts[:,0]
yi = pts[:,1]
zi = pts[:,2]

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
plt.savefig('ball_boundary.png')