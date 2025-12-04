# Importing PyTorch
import torch
import time
import numpy as np

# Print the version of PyTorch
print("Pytorch Version:   ", torch.__version__)

# Introduction to Tensors
# Scalar
scalar = torch.tensor(7)
print("\n", scalar)
print("Scalar Item:  ", scalar.item())
print("Scalar Dimensions: ", scalar.ndim)
print("Scalar Type: ", type(scalar))
print("Scalar DataType: ", scalar.dtype)

# Vector
vector1 = torch.tensor([7, 7])
vector2 = torch.tensor((7,7))

print("\n", vector1, vector2)

print("Vector1[0] Items:", vector1[0].item())
print("Vector2[0] Items:", vector2[0].item())
print("Vector1 Dimensions: ", vector1.ndim)
print("Vector2 Dimensions: ", vector2.ndim)
print("Vector1 Shape: ", vector1.shape)
print("Vector2 Shape: ", vector2.shape)
print("Vector1 DataType: ", vector1.dtype)

# Matrix
MATRIX = torch.tensor([[7,8], [9,10]])
print("\n", MATRIX)
print("MATRIX Dimension: ", MATRIX.ndim)
print("MATRIX Shape: ", MATRIX.shape)
print("Matrix[0] Dimensions: ", MATRIX[0].ndim)
print("Matrix[1] Dimensions: ", MATRIX[0].ndim)
print("Matrix [0] Shape: ", MATRIX[0].shape)
print("Matrix [1] Shape: ", MATRIX[1].shape)
print("Matrix DataType: ", MATRIX.dtype)

# Tensor
TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
print("\n", TENSOR)
print("TENSOR Dimensions: ", TENSOR.ndim)
print("TENSOR Shape: ", TENSOR.shape)
print("TENSOR[0][0]", TENSOR[0][0])
print("TENSOR[0][1]", TENSOR[0][1])
print("TENSOR[0][2]", TENSOR[0][2])
print("Tensor DataType", TENSOR.dtype)

# Random Tensors
random_tensor = torch.rand(size=(1,3,4))
print("\nRandom Tensor: ", random_tensor)
print("Random Tensor Dimension", random_tensor.ndim)
print("Random Tensor Shape: ", random_tensor.shape)
print("Random Tensor[0] Dimension", random_tensor[0].ndim)
print("Random Tensor[0] Shape: ", random_tensor[0].shape)
print("Random Tensor[0][0] Dimension", random_tensor[0][0].ndim)
print("Random Tensor[0][0] Shape: ", random_tensor[0][0].shape)
print("Random Tensor DataType", random_tensor.dtype)

# Zero Tensors
zero = torch.zeros(size=(3,4))
print("\nZero Tensor: ", zero)
print("Zero Tensor Dimension: ", zero.ndim)
print("Zero Tensor Shape: ", zero.shape)
print("Zero Tensor DataType: ", zero.dtype)

# Ones Tensor
ones = torch.ones(size=(3,4))
print("\nOne Tensor: ", ones)
print("One Tensor Dimension: ", ones.ndim)
print("One Tensor Shape: ", ones.shape)
print("One Tensor Datatype: ", ones.dtype)

# Creating a range of tensors
one_to_ten = torch.arange(start=1, end=11, step=1)
print("\nRange Tensor: ", one_to_ten)

# Creating tensor like from existing tensors
zero_like_tensor = torch.zeros_like(input=one_to_ten)
print("\nZero Like Tensor: ", zero_like_tensor)

one_like_tensor = torch.ones_like(input=MATRIX)
print("\nOne Like Tensor: ", one_like_tensor)

# Tensor Datatypes
# The default data type in pytorch is torch.float32
# The most common errors encountered in deep leanring are:
#     a. Tensors not of right datatype
#     b. Tensors not of right shape
#     c. Tensors not on the right device
float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None,
                               device=None,
                               requires_grad=True)
print("\nFloat 32 Tensor: ", float_32_tensor)
print("Float 32 Tensor Dimension: ", float_32_tensor.ndim)
print("Float 32 Tensor Shape: ", float_32_tensor.shape)
print("Float 32 Tensor DataType: ", float_32_tensor.dtype)
print("Float 32 Tensor Device: ", float_32_tensor.device)

# Creating new data type from existing
float_16_tensor = float_32_tensor.type(torch.float16)
print("\nFloat 16 Tensor: ", float_16_tensor)
print("Float 16 Tensor Dimension: ", float_16_tensor.ndim)
print("Float 16 Tensor Shape: ", float_16_tensor.shape)
print("Float 16 Tensor DataType: ", float_16_tensor.dtype)
print("Float 16 Tensor Device: ", float_16_tensor.device)

# Getting attributes of a tensor
some_tensor = torch.rand([3,4],dtype=torch.float64)
print("\nSome Tensor: ", some_tensor)
print("Some Tensor Dimension: ", some_tensor.ndim)
print("Some Tensor Shape: ", some_tensor.shape)
print("Some Tensor DataType: ", some_tensor.dtype)
print("Some Tensor Device: ", some_tensor.device)

# Tensor Operations: Addition, Subtraction, Multiplication, Division
tensor1 = torch.rand([10])
tensor2 = torch.rand([10])
print("\nTensor 1: ", tensor1)
print("Tensor 2: ", tensor2)

# Adding something to tensor
print("Adding 10 to Tensor1: ", tensor1+10)
print("Adding two Tensors: ", tensor1+tensor2)
print("Adding two Tensors (with add()): ", tensor1.add(tensor2))

# Subtracting something to tensor
print("\nSubtracting 10 to Tensor1: ", tensor1-10)
print("Subtracting two Tensors: ", tensor1-tensor2)
print("Subtracting two Tensors (with sub()): ", tensor1.sub(tensor2))

# Multiplying something to tensor
print("\nMultiply 10 to Tensor1: ", tensor1*10)
print("Multiplying two Tensors (Elementwise Multiplication): ", tensor1*tensor2)
print("Multiplying two Tensors (Elementwise Multiplication with mul()): ", tensor1.mul(tensor2))
print("Multiplying two Tensors (Matrix Multiplication or Dot Product): ", tensor1.matmul(tensor2))
print("Multiplying two Tensors (Matrix Multiplication or Dot Product using @): ", tensor1 @ tensor2)

# Dividing something to tensor
print("\nDivide 10 to Tensor1: ", tensor1/10)
print("Divide two Tensors: ", tensor1/tensor2)

# Time Difference between classical and inbuild functions
start_time = time.time()
print("\nTensor1 * Tensor2: ", tensor1*tensor2)
print("Tensor1 * Tensor2 Duration: ", time.time() - start_time)

start_time = time.time()
print("\nTensor1.mul(Tensor2): ", tensor1.mul(tensor2))
print("Tensor1.mul(Tensor2) Duration: ", time.time() - start_time)

# Transposing a Tensor
tensor1 = torch.rand([4,5])
print("\nOriginal Tensor1: ", tensor1)
print("Transposed Tensor1: ", tensor1.T)
print("Original Tensor1 Shape", tensor1.shape)
print("Transposed Tensor1 Shape: ", tensor1.T.shape)

# Performing min/max/mean and sum of tensor
x = torch.arange(start=1, end=11, step=1)
print("\n x Tensor: ", x)
print("Min of x Tensor (torch.min()): ", torch.min(x).item())
print("Min of x Tensor (x.min()): ", x.min().item())

print("Max of x Tensor (torch.max()): ", torch.max(x).item())
print("Max of x Tensor (x.max()): ", x.max().item())

print("Sum of x Tensor (torch.sum()): ", torch.sum(x, dtype=torch.float32).item())
print("Sum of x Tensor (x.sum()): ", x.sum(dtype=torch.float32).item())

print("Mean of x Tensor (torch.mean()): ", torch.mean(x, dtype=torch.float32).item())
print("Mean of x Tensor (x.mean()): ", x.mean(dtype=torch.float32).item())

# Performing positional min/max
print("\nFind the position of the tensor that has minimum value in the tensor", x.argmin())
print("\nFind the position of the tensor that has maximum value in the tensor", x.argmax())

# Reshaping, stacking, squeezing and unsqueezing tensors
# Reshaping a Tensor
x = torch.arange(start=100, end=300, step=10)
print("\n x Tensor: ", x)
print("x Tensor DataType: ", x.dtype)
print("x Tensor Shape: ", x.shape)

x_reshaped = x.reshape([5,4])
print("x_Reshaped: ", x_reshaped)
x_reshaped = x.reshape([20,1])
print("x_Reshaped: ", x_reshaped)
x_reshaped = x.reshape([4,5])
print("x_Reshaped: ", x_reshaped)

# Change the View of the tensor now
# (View changes shape but shares the same memory as that of the original tensor)
# (Changing view elements changes the original tensor as well because they share same memory)
z = x.view(5,4)
print("z Tensor: ", z)
print("z Tensor View Shape: ", z.shape)
# Change one of the value in z-tensor
z[1,0] =11
print("x Tensor After Modifying z: ", z)

# Stacking tensors together
x = torch.tensor([1,2,3,4,5.])
print("\nx Original Tensor: ", x)
x_stacked = torch.stack([x, x, x], dim=0)
print("x_Stacked Tensor (Horizontally): ", x_stacked)
x_stacked = torch.stack([x, x, x], dim=1)
print("x_Stacked Tensor (Vertically): ", x_stacked)

# Squeeze all the 1 dimensions out of the tensor
x = torch.rand([1,2,1,1,4], dtype=torch.float32)
print("\n x Tensor: ", x)
print("x Tensor Shape: ", x.shape)
print("x Tensor DataType: ", x.dtype)
print("x Tensor Device: ", x.device)
x_squeezed = x.squeeze()
print("x Tensor Squeezed: ", x_squeezed)
print("x Tensor Squeezed Shape: ", x_squeezed.shape)

# Unsqueeze a tensor to add 1 more dimension
# dim=0: Adds 0th dimension as 1
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print("x Tensor Unsqueezed: ", x_unsqueezed)
print("x Tensor Unsqueezed Shape: ", x_unsqueezed.shape)
# dim=0: Adds second dimension as 1
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print("x Tensor Unsqueezed: ", x_unsqueezed)
print("x Tensor Unsqueezed Shape: ", x_unsqueezed.shape)

# Permute: Creates a view of the tensor with desired ordering of dimensions
x = torch.rand(size=[1, 2, 3, 1, 4], dtype=torch.float32)
print("\n x Tensor: ", x)
# Give the dimensions in terms of index for re-aranging
x_permuted = x.permute(1, 2, 4, 0, 3)
print("x_Permuted: ", x_permuted)
print("x_Permuted Shape: ", x_permuted.shape)
# Changing a value in x-permuted
x_permuted[0][0][0][0][0] = 5
print("x Tensor: ", x)

# Indexing in tensor
# Using :, we can retain a whole dimension of tensor while indexing
x = torch.rand(size=[1, 5, 4, 6], dtype=torch.float32)
print("\n x Tensor: ", x)
#  Getting x[0]
print("x[0]: ", x[0])
print("x[0][0]: ", x[0,0])
print("x[0,0,0]: ", x[0,0,0])
print("x[0,0,0,0]: ", x[0,0,0,0])
print("x[0,0,0,0] Item: ", x[0,0,0,0].item())

# Keep first dimension intact and return the rest x[0,0,0]
print("x with First Dimension Intact: ", x[:, 0, 0, 0])
print("x with First Two Dimension Intact: ", x[:, :, 0, 0])

# Numpy and Pytorch Conversions
np_array = np.random.randn(4, 4)
print("\n np_array: ", np_array)
print("\n np_array Dimensions: ", np_array.ndim)
print("\n np_array Shape: ", np_array.shape)
print("\n np_array DataType: ", np_array.dtype)
# Converting np_array to tensor
np_tensor = torch.from_numpy(np_array).type(torch.float32)
print("\n np_tensor: ", np_tensor)
print("\n np_tensor DataType: ", np_tensor.dtype)
# Converting a tensor to numpy array
np_array = x[0][0][0].numpy()
print("\n np_array: ", np_array)
print("\n np_array DataType: ", np_array.dtype)

# Pytorch Reproducibility
# Using torch.manual_seed(), we can generate custom randomness
random_tensor_A = torch.randn(2, 2)
random_tensor_B = torch.randn(2, 2)
print("\n***Without Manual Seed***")
print("Random Tensor A: ", random_tensor_A)
print("Random Tensor B: ", random_tensor_B)
print("A == B: ", random_tensor_A==random_tensor_B)
# Setting manual seed now for A only
torch.manual_seed(40)
random_tensor_A = torch.randn(2, 2)
random_tensor_B = torch.randn(2, 2)
print("\n***With Manual Seed for A Only***")
print("Random Tensor A: ", random_tensor_A)
print("Random Tensor B: ", random_tensor_B)
print("A == B: ", random_tensor_A==random_tensor_B)
# Setting manual seed now for both A and B
torch.manual_seed(40)
random_tensor_A = torch.randn(2, 2)
torch.manual_seed(40)
random_tensor_B = torch.randn(2, 2)
print("\n***With Manual Seed for A Only***")
print("Random Tensor A: ", random_tensor_A)
print("Random Tensor B: ", random_tensor_B)
print("A == B: ", random_tensor_A==random_tensor_B)

#  Check for GPU Availability
print("\nIs CUDA Available: ", torch.cuda.is_available())

# Write device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
number_of_devices = torch.cuda.device_count()
print("Hardware Device: ", device)
print("Number of GPUs: ", number_of_devices)

# Shifting a tensor from CPU to GPU
cpu_tensor = torch.rand(size=(3,4), dtype=torch.float16)
gpu_tensor = cpu_tensor.to(device=device)
print("GPU Tensor Device : ", gpu_tensor.device)

# Converting a GPU tensor back to CPU tensor for numpy
# Numpy doesn't work for GPU tensors, so any numpy conversion needs only the CPU Tensor
cpu_tensor = gpu_tensor.cpu().numpy()
print("\n Transformed GPU to CPU Tensor: ", cpu_tensor)













