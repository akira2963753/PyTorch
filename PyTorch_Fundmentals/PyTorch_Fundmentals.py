import torch
# Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.shape) # No scalar don't use item()

#Vector
vector = torch.tensor([7,7])
print(vector)
# Check the number of dimensions of vector
print(vector.ndim)
# Check shape(Size) of vector
print(vector.shape)

# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape) # torch.Size [1,3,3]

# Create a random tensor of size (3,4) In Tensor Random Value 由 0 ~ 1
import torch
random_tensor = torch.rand(size=(3,4))
print(random_tensor)
print(random_tensor.dtype)
print(random_tensor.shape)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224,224,3))
print(random_image_size_tensor)
print(random_image_size_tensor.dtype)
print(random_image_size_tensor.shape)

#%%
import torch
# Create a tensor of all zeros
zeros = torch.zeros(size=(3,4))
print(zeros,zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3,4),dtype=torch.int)
print(ones,ones.dtype)

# Use torch.arange(), torch.range() is deprecated 
zero_to_ten = torch.arange(start=0, end=10, step=1) 
print(zero_to_ten)

# Can also creat a tensor of zero similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)

# Can Copy
zero_to_ten_copy = zero_to_ten.clone()
print(zero_to_ten_copy)

#%%
import torch
# Default Datatype for tensors is float32
float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype=None,device=None,requires_grad=False)
print(float_32_tensor.shape,float_32_tensor.dtype,float_32_tensor.device)

float_16_tensor = torch.tensor([3.0,6.0,9.0],dtype=torch.float16)
print(float_16_tensor.dtype)

#%%
import torch
# Create a tensor
some_tensor = torch.rand(3,4)

# Find out details about it
print(some_tensor)
print(f"Shap of tensor :  {some_tensor.shape}")
print(f"Datatype of tensor : {some_tensor.dtype}")
print(f"Device tensor is stored on : {some_tensor.device}")

#%%
import torch
# Create a tensor of values and add a number to it
tensor = torch.tensor([1,2,3])
print(tensor+10)
print(tensor*10)
print(tensor-10)
print(torch.multiply(tensor,10))
print(tensor*tensor)
print(tensor)

#%%
import torch
tensor = torch.tensor([1,2,3]) 
print(tensor.shape)
print(tensor*tensor)

# Matrix Multiplication -> matmul
print(torch.matmul(tensor,tensor))
print(tensor@tensor)

#%%
import torch
tensor_A = torch.tensor([[1,2],
                        [3,4],
                        [5,6]],dtype=torch.float32,requires_grad=False)
tensor_B = torch.tensor([[7,8,9],
                        [10,11,12]],dtype=torch.float32)
print(tensor_A@tensor_B)

#Transpose tensor
print(tensor_A.T@tensor_B.T)

# torch.mm is a shortcut for matmul
print(torch.mm(tensor_A.T, tensor_B.T))

# http://matrixmultiplication.xyz/ Can use Matrix Multiplication

# Since the linear  layer starts with a random weights matrix
# Let's make it reproducible
#%%
import torch
tensor_A = torch.tensor([[1,2],
                        [3,4],
                        [5,6]],dtype=torch.float32,requires_grad=False)
torch.manual_seed(42)
# This uses matrix multiplication W is 6x2
linear = torch.nn.Linear(in_features=2,out_features=6) 

# Don't require gradient
linear.weight.requires_grad_(False)
linear.bias.requires_grad_(False)

x = tensor_A # X is 3x2
output = linear(x)
print(f"Bias Parameter : {linear.bias}")
print(f"Bias Value : {linear.bias.data}")
print(f"\nInput shape : {x.shape}\n")
print(f"Output : \n{output}\n\nOutput shape : {output.shape}")

#%%
import torch
x = torch.arange(0,100,10)
print(x)
print(f"Minimun : {x.min()}")
print(f"Maximun : {x.max()}")

# print(f"Mean: {x.mean()}") # this will error
# won't work without float datatype , because needing divide
print(f"Mean: {x.type(torch.float32).mean()}")
print(f"Sum: {x.sum()}")

# Returns index of max and min values
print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")

#%%
import torch

# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)

# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.dtype)

# Create an int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8.dtype)

#%%
import torch
# Create a tensor
x = torch.arange(1.,8.)
print(x,x.shape)

# Add an extra dimension
x_reshaped = x.reshape(1,7)
print(x_reshaped,x_reshaped.shape)

z = x.view(1,7)
print(z,z.shape)

# : -> 代表把所有行，0 -> 第0列
z[:,0] = 5
print(z,x)

x_stacked = torch.stack([x,x,x,x] , dim=0)
print(x_stacked)

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
#%%
import torch
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")

#%%
# Create a tensor 
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x,x.shape)

# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")

print(x[:, 0])
print(x[:, :, 1])
print(x[:, 1, 1])
print(x[0, 0, :])

#%%
import torch
import numpy as np

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)
print(array)
print(tensor)

array = array + 1
print(array)
print(tensor)
#%%
import torch
import numpy as np

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor)
print(numpy_tensor)

tensor = tensor + 1
print(tensor)
print(numpy_tensor)

#%%
import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
print(random_tensor_A == random_tensor_B)

#%%
import torch
import random

# # Set the random seed
RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called 
# Without this, tensor_D would be different to tensor_C 
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
print(random_tensor_C == random_tensor_D)

# %%
import torch
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())
#%%
import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
print(tensor_on_gpu)