from config import setup_jax_environment

# configure before importing jax
setup_jax_environment()

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import pmap, local_device_count
from jax.lax import pmean
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax import jax_utils
import optax
import numpy as np
import functools as ft
import matplotlib.pyplot as plt
import einops
import os
import gzip
import struct
import array
import urllib.request
from typing import Callable, Tuple, Any, Dict
import diffrax as dfx
from tqdm.auto import tqdm
import time
import gc

# Model definition - Imported from previous implementation
class MLP(nn.Module):
    features: list
    
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(features=feat)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.features[-1])(x)
        return x


class MixerBlock(nn.Module):
    num_patches: int
    hidden_size: int
    mix_patch_size: int
    mix_hidden_size: int
    
    def setup(self):
        self.patch_mixer = MLP(
            features=[self.mix_patch_size, self.num_patches],
            name="patch_mixer"
        )
        self.hidden_mixer = MLP(
            features=[self.mix_hidden_size, self.hidden_size],
            name="hidden_mixer"
        )
        self.norm1 = nn.LayerNorm(epsilon=1e-5, name="norm1")
        self.norm2 = nn.LayerNorm(epsilon=1e-5, name="norm2")
    
    def __call__(self, y, *, train=True):
        # First mixing operation (along patches dimension)
        # print(f'input y-shape: {y.shape}')
        mixed_patches = jax.vmap(self.patch_mixer)(self.norm1(y)) 
        # print(f'mixed shape: {mixed_patches.shape}')
        y = y + mixed_patches
        
        # Transpose and then apply second mixing operation
        y = einops.rearrange(y, "c p -> p c")
        mixed_hidden = jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = y + mixed_hidden
        
        # Return to original shape
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(nn.Module):
    img_size: Tuple[int, int, int]  # (channels, height, width)
    patch_size: int
    hidden_size: int
    mix_patch_size: int
    mix_hidden_size: int
    num_blocks: int
    t1: float
    
    def setup(self):
        # self.params = [('img_size', self.img_size), ('patch_size', self.patch_size), ('hidden_size',self.hidden_size), 
                        #    ('mix_patch_size',self.mix_patch_size), ('mix_hidden_size', self.mix_hidden_size), ('num_blocks', self.num_blocks)]
        input_size, height, width = self.img_size
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)
        
        self.conv_in = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="conv_in"
        )
        
        self.conv_out = nn.ConvTranspose(
            features=input_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="conv_out"
        )
        
        # Create blocks
        self.blocks = [
            MixerBlock(
                num_patches=self.num_patches,
                hidden_size=self.hidden_size,
                mix_patch_size=self.mix_patch_size,
                mix_hidden_size=self.mix_hidden_size,
                name=f"block_{i}"
            )
            for i in range(self.num_blocks)
        ]
        
        self.norm = nn.LayerNorm(epsilon=1e-5, name="norm")
    
    def __call__(self, t, y, *, train=True):
        t = jnp.array(t / self.t1)
        _, height, width = y.shape
        t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        y = jnp.concatenate([y, t])
        
        # print(f'y after concatenation: {y.shape}')
        
        # Apply initial convolution
        y = jnp.transpose(self.conv_in(jnp.transpose(y)))
        
        
        # print(f'y after convolution: {y.shape}')
        _, patch_height, patch_width = y.shape
        
        # Reshape for mixer blocks
        y = einops.rearrange(y, "c h w -> c (h w)")
        # print(f'y after rearrange: {y.shape}')
        
        # Apply mixer blocks
        for block in self.blocks:
            y = block(y, train=train)
        
        # Apply normalization
        y = self.norm(y)
        
        # Reshape back for final convolution
        y = einops.rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        
        # Apply output convolution
        return self.conv_out(y)


# Data loading functions
def mnist():
    """Download and load the MNIST dataset."""
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"
    
    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")
    
    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)


def dataloader(data, batch_size, *, key):
    """Dataloader with support for device sharding."""
    num_devices = jax.local_device_count()
    # Ensure batch size is divisible by number of devices
    per_device_batch_size = batch_size // num_devices
    total_batch_size = per_device_batch_size * num_devices
    
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = total_batch_size
        
        while end <= dataset_size:
            batch_perm = perm[start:end]
            batch_data = data[batch_perm]
            # Reshape for device sharding: (batch_size, ...) -> (num_devices, per_device_batch_size, ...)
            batch_data = batch_data.reshape(num_devices, per_device_batch_size, *batch_data.shape[1:])
            yield batch_data
            start = end
            end = start + total_batch_size


def batch_loss_fn(model_apply, params, batch, t1, key):
    """Compute loss for a batch of data."""
    # Define noise schedule functions inside batch loss
    int_beta = lambda t: t # linear schedule
    weight = lambda t: 1 - jnp.exp(-int_beta(t)) # Upweight region near t=0
    
    batch_size = batch.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    
    # Low-discrepancy sampling over t
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    
    # Vectorize the loss computation with integrated functions
    def single_loss_fn(data, t, key):
        """Compute loss for a single timestep."""
        mean = data * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, data.shape)
        y = mean + std * noise
        
        # Apply model
        pred = model_apply({"params": params}, t, y)
        
        return weight(t) * jnp.mean((pred + noise / std) ** 2)
    
    # Vectorize the loss computation
    loss_fn = jax.vmap(single_loss_fn)
    
    return jnp.mean(loss_fn(batch, t, losskey))


# Distributed training functions
def create_train_state(rng, model, learning_rate):
    """Create initial TrainState for the model."""
    params = model.init(rng, jnp.zeros(()), jnp.zeros((1, 28, 28)))["params"]
    tx = optax.adabelief(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@ft.partial(pmap, axis_name='devices')
def train_step(state, batch, t1_array, rng):
    """Distributed training step using pmap.
    
    Args:
        state: The replicated TrainState.
        batch: The data batch, already shared across devices.
        t1_array: Array containing t1 value, replicated across devices.
        rng: Random key, replicated across devices. 
    
    """
    t1 = t1_array[0]
    
    def loss_fn(params):
        loss = batch_loss_fn(state.apply_fn, params, batch, t1, rng)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    
    # Average gradients across all devices
    grads = pmean(grads, axis_name='devices')
    
    # Update parameters using optimizer
    state = state.apply_gradients(grads=grads)
    
    # Update random key
    new_rng = jr.fold_in(rng, 0)
    
    return state, loss, new_rng


@ft.partial(pmap, axis_name='devices')
def generate_sample(state, data_shape, dt0_array, t1_array, rng):
    """Generate sample on each device."""
    
    dt0 = dt0_array[0]
    t1 = t1_array[0]
    
    int_beta = lambda t: t
    
    
    def drift(t, y, args):
        # Calculate derivative of int_beta
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        model_output = state.apply_fn({"params": state.params}, t, y)
        return -0.5 * beta * (y + model_output)
    
    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(rng, data_shape)
    
    # Solve ODE from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]


def print_gpu_memory():
    """Print memory usage for each GPU."""
    try:
        import subprocess
        import re
        
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits']
        ).decode('utf-8')
        
        print("\nGPU Memory Usage:")
        for line in result.strip().split('\n'):
            idx, used, total = map(int, re.findall(r'\d+', line))
            print(f"GPU {idx}: {used} MB / {total} MB ({used/total*100:.1f}%)")
    except Exception as e:
        print(f"Unable to query GPU memory: {e}")


def measure_model_memory_impact(model, input_shapes, rng):
    """Measure the memory impact of a model."""
    # Create dummy inputs
    t_dummy = jnp.zeros(())
    y_dummy = jnp.zeros(input_shapes)
    
    print("\nMeasuring model memory impact:")
    print_gpu_memory()
    
    # Initialize model and measure memory
    print("Initializing model...")
    params = model.init(rng, t_dummy, y_dummy)["params"]
    print_gpu_memory()
    
    # Forcing a forward pass and measuring memory
    print("Running forward pass...")
    _ = model.apply({"params": params}, t_dummy, y_dummy)
    print_gpu_memory()
    
    # Calculating parameter count
    param_count = sum(np.prod(p.shape) for p in jax.tree_leaves(params))
    print(f"\nModel parameter count: {param_count:,} parameters")
    param_bytes = param_count * 4  # Assuming float32
    print(f"Model parameter size: {param_bytes / (1024**2):.2f} MB")
    
    return params

def configure_jax_memory(
    preallocate=False,
    memory_fraction=0.4,
    use_platform_allocator=True,
    force_device_placement=False
):
    """Configure JAX for more memory-efficient operation."""
    # Disable preallocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' if preallocate else 'false'
    
    # Set memory fraction
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    
    # Use platform allocator (can reduce memory fragmentation)
    if use_platform_allocator:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Try to prevent device 0 from being overused
    if not force_device_placement:
        os.environ['XLA_PYTHON_CLIENT_DEVICE_PREALLOCATE'] = 'false'
    
    # Limit number of CompileOptions for XLA
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    
    # Print the configuration
    print("JAX Memory Configuration:")
    print(f"  Preallocate: {preallocate}")
    print(f"  Memory Fraction: {memory_fraction}")
    print(f"  Platform Allocator: {use_platform_allocator}")
    print(f"  Force Device Placement: {force_device_placement}")
    
    # Clear any existing compilation cache
    jax.clear_caches()
    
    # Give the system a moment to stabilize
    time.sleep(1)
    print_gpu_memory()

def initialize_model_distributed(model, input_shapes, rng, num_devices):
    """Initialize model parameters in a memory-balanced way across devices."""
    # Create dummy inputs
    t_dummy = jnp.zeros(())
    y_dummy = jnp.zeros(input_shapes)
    
    # Create separate RNGs for each device
    rngs = jr.split(rng, num_devices)
    
    # Define a device-local init function
    def init_on_device(rng):
        return model.init(rng, t_dummy, y_dummy)["params"]
    
    # Map the init function across devices to balance memory
    params_list = pmap(init_on_device)(rngs)
    
    # Use params from first device (they're all identical)
    return jax.tree_map(lambda x: x[0], params_list)

def memory_efficient_dataloader(data_fn, batch_size, *, key, buffer_size=10000):
    """Memory-efficient dataloader that loads data in chunks.
    
    Args:
        data_fn: Function that returns a data chunk when called with (start, end) indices
        batch_size: Batch size for training
        key: Random key for permutation
        buffer_size: Size of the buffer window to permute
    """
    num_devices = jax.local_device_count()
    per_device_batch_size = batch_size // num_devices
    total_batch_size = per_device_batch_size * num_devices
    
    # Get dataset size without loading all data
    dataset_size = data_fn.get_size()
    
    # Use a buffer instead of permuting the entire dataset
    buffer_size = min(buffer_size, dataset_size)
    
    while True:
        # Create a buffer of indices to permute
        key, buffer_key = jr.split(key)
        buffer_indices = jr.randint(
            buffer_key, 
            (buffer_size,), 
            minval=0, 
            maxval=dataset_size
        )
        
        # Permute the buffer for better mixing
        key, perm_key = jr.split(key)
        buffer_perm = jr.permutation(perm_key, buffer_indices)
        
        # Load and yield batches from the buffer
        for start in range(0, buffer_size, total_batch_size):
            end = min(start + total_batch_size, buffer_size)
            if end - start < total_batch_size:
                # Not enough samples in buffer, get a new one
                break
                
            # Get indices for this batch
            batch_indices = buffer_perm[start:end]
            
            # Load only the data for this batch
            batch_data = data_fn(batch_indices)
            
            # Reshape for device sharding
            batch_data = batch_data.reshape(num_devices, per_device_batch_size, *batch_data.shape[1:])
            yield batch_data


class MNISTDataProvider:
    """Memory-efficient data provider for MNIST."""
    
    def __init__(self, normalize=True, seed=42):
        """Initialize the data provider.
        
        Args:
            normalize: Whether to normalize the data
            seed: Random seed for initialization
        """
        self.normalize = normalize
        self.rng = np.random.RandomState(seed)
        
        # Load metadata but not the actual data
        self._init_metadata()
        
    def _init_metadata(self):
        """Initialize metadata like data mean, std without loading all data."""
        filename = "train-images-idx3-ubyte.gz"
        url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
        target_dir = os.getcwd() + "/data/mnist"
        target = f"{target_dir}/{filename}"
        
        if not os.path.exists(target):
            os.makedirs(target_dir, exist_ok=True)
            urllib.request.urlretrieve(f"{url_dir}/{filename}", target)
            print(f"Downloaded {url_dir}/{filename} to {target}")
        
        # Just read the header to get dataset size
        with gzip.open(target, "rb") as fh:
            _, self.size, self.rows, self.cols = struct.unpack(">IIII", fh.read(16))
            self.data_shape = (1, self.rows, self.cols)
            
            # Calculate statistics using a sample to avoid loading all data
            sample_size = min(10000, self.size)
            sample_indices = self.rng.choice(self.size, sample_size, replace=False)
            
            # Seek to each index and read only those samples
            sample_data = []
            for idx in sorted(sample_indices):  # Sort for sequential access
                fh.seek(16 + idx * self.rows * self.cols)
                img_bytes = fh.read(self.rows * self.cols)
                img = np.array(array.array("B", img_bytes), dtype=np.uint8)
                img = img.reshape(1, self.rows, self.cols)
                sample_data.append(img)
            
            sample_data = np.concatenate(sample_data, axis=0)
            
            # Calculate statistics
            self.data_mean = float(np.mean(sample_data))
            self.data_std = float(np.std(sample_data))
            self.data_min = float(np.min(sample_data))
            self.data_max = float(np.max(sample_data))
        
        print(f"MNIST metadata initialized: {self.size} images of shape {self.data_shape}")
        print(f"Stats: mean={self.data_mean:.2f}, std={self.data_std:.2f}, min={self.data_min}, max={self.data_max}")
        
        # Save the file path for later use
        self.file_path = target
    
    def get_size(self):
        """Get the dataset size."""
        return self.size
    
    def get_shape(self):
        """Get the data shape."""
        return self.data_shape
    
    def get_stats(self):
        """Get dataset statistics."""
        return {
            "mean": self.data_mean,
            "std": self.data_std,
            "min": self.data_min,
            "max": self.data_max
        }
        
    def __call__(self, indices):
        """Load specific indices from the dataset.
        
        Args:
            indices: Array of indices to load
            
        Returns:
            Array of data samples
        """
        data = []
        
        # Open the file and read only the requested indices
        with gzip.open(self.file_path, "rb") as fh:
            for idx in indices:
                # Seek to the position of this index in the file
                offset = 16 + idx * self.rows * self.cols
                fh.seek(offset)
                
                # Read the image data
                img_bytes = fh.read(self.rows * self.cols)
                img = np.array(array.array("B", img_bytes), dtype=np.float32)
                img = img.reshape(self.data_shape)
                
                if self.normalize:
                    img = (img - self.data_mean) / self.data_std
                
                data.append(img)
        
        # Stack the data into a batch
        return jnp.array(np.stack(data))


# def main(
#     patch_size=4,
#     hidden_size=64,
#     mix_patch_size=512,
#     mix_hidden_size=512,
#     num_blocks=4,
#     t1=10.0,
#     num_steps=100_000,
#     lr=3e-4,
#     batch_size=512,
#     print_every=10_000,
#     checkpoint_every=50_000,
#     dt0=0.1,
#     sample_size=10,
#     checkpoint_dir="./checkpoints",
#     seed=5678,
#     memory_fraction=0.8,
# ):
#     """Main training function with multi-GPU support."""
    
    
#     os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
#     os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
#     # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
#     # os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1 --xla_gpu_deterministic_ops=true'
    
#     # configure_jax_memory(
#     #     preallocate=False,
#     #     memory_fraction=0.4,
#     #     use_platform_allocator=True
#     # )
    
#     # Check available devices
#     num_devices = jax.local_device_count()
#     print(f"Using {num_devices} devices with memory fraction {memory_fraction}")
#     print_gpu_memory()
    
    
#     # Adjust batch size to be divisible by number of devices
#     if batch_size % num_devices != 0:
#         orig_batch_size = batch_size
#         batch_size = (batch_size // num_devices) * num_devices
#         print(f"Adjusted batch size from {orig_batch_size} to {batch_size} to be divisible by {num_devices}")
    
#     # Set up RNG keys - one per device
#     master_rng = jr.PRNGKey(seed)
#     model_rng, train_rng, data_rng, sample_rng = jr.split(master_rng, 4)
    
#     # Create per-device RNG keys
#     train_rngs = jr.split(train_rng, num_devices)
#     sample_rngs = jr.split(sample_rng, num_devices)
    
#     # Load and normalize data
#     print("Loading MNIST dataset...")
#     data_provider = MNISTDataProvider(normalize=True, seed=seed)
#     data_shape = data_provider.get_shape()
#     data_stats=  data_provider.get_stats()
    
#     data_iter = memory_efficient_dataloader(
#         data_provider,
#         batch_size,
#         key=data_rng,
#         buffer_size=5000
#     )
    
#     # Initialize model
#     print("Initializing model...")
    
    
#     # Measure memory before model initialization
#     print("Memory before model initialization")
#     print_gpu_memory()
    
#     # Try to initialize model on CPU to avoid GPU spike
#     with jax.default_device(jax.devices('cpu')[0]):
#         try:
#             model = Mixer2d(
#                 img_size=data_shape,
#                 patch_size=patch_size,
#                 hidden_size=hidden_size,
#                 mix_patch_size=mix_patch_size,
#                 mix_hidden_size=mix_hidden_size,
#                 num_blocks=num_blocks,
#                 t1=t1
#             )
            
#             # Initialize parameters on CPU
#             cpu_t_dummy = jnp.zeros(())
#             cpu_y_dummy = jnp.zeros(data_shape)
#             cpu_params = model.init(model_rng, cpu_t_dummy, cpu_y_dummy)["params"]
            
#             # Create optimizer
#             tx = optax.adabelief(lr)
            
#             # Create initial state on CPU
#             cpu_state = train_state.TrainState.create(
#                 apply_fn=model.apply,
#                 params=cpu_params,
#                 tx=tx
#             )
#             # Now transfer to devices evenly
#             print("Transferring parameters to devices...")
#             gpu_state = jax.device_put(cpu_state, device=jax.devices('gpu')[0])
            
#             # This will still use default device for compilation but avoid
#             # parameter replication on a single device
#             state = jax_utils.replicate(gpu_state)
            
#             print("Memory after CPU initialization and transfer:")
#             print_gpu_memory()
        
#         except Exception as e:
#             print(f"CPU initialization failed: {e}. Falling back to GPU initialization")
#             # If CPU initialization fails, try distributed initialization
#             model = Mixer2d(
#                 img_size=data_shape,
#                 patch_size=patch_size,
#                 hidden_size=hidden_size,
#                 mix_patch_size=mix_patch_size,
#                 mix_hidden_size=mix_hidden_size,
#                 num_blocks=num_blocks,
#                 t1=t1
#             )
            
#             # Use balanced device initialization 
#             t_dummy = jnp.zeros(())
#             y_dummy = jnp.zeros(data_shape)
            
#             # Create separate RNGs for each device
#             init_rngs = jr.split(model_rng, num_devices)
            
#             # Define a device-local init function
#             @pmap
#             def distributed_init(rng):
#                 return model.init(rng, t_dummy, y_dummy)["params"]
            
#             # Initialize in parallel across devices
#             params_replicated = distributed_init(init_rngs)
#             params = jax.tree.map(lambda x: x[0], params_replicated)
            
#             # Create optimizer
#             tx = optax.adabelief(lr)
            
#             # Create state
#             state = train_state.TrainState.create(
#                 apply_fn=model.apply,
#                 params=params,
#                 tx=tx
#             )
            
#             # Replicate for training
#             state = jax_utils.replicate(state)
            
    
    
#     # After initialization to reduce GPU mem 
#     jax.clear_caches()
#     gc.collect()
    
#     print("Memory after model initialization:")
#     print_gpu_memory()
    
#     # Create and distribute training state
#     # print("Creating training state...")
#     # state = create_train_state(model_rng, model, lr)
    
#     # # Create optimizer separately
#     # tx = optax.adabelief(lr)
    
#     # # Create train state with pre-initialized params
#     # state = train_state.TrainState.create(
#     #     apply_fn=model.apply,
#     #     params=params,
#     #     tx=tx
#     # )
    
#     # Replicate state across devices
#     # state = jax_utils.replicate(state)
    
#     # Create arrays for scalar parameters and replicate across devices
#     t1_array = jnp.array([t1])
#     dt0_array = jnp.array([dt0])
#     t1_replicated = jax_utils.replicate(t1_array)
#     dt0_replicated = jax_utils.replicate(dt0_array)
    
#     # Create checkpoint directory
#     checkpoint_dir = os.path.abspath(checkpoint_dir)
#     print(f"Using absolute checkpoint path: {checkpoint_dir}")
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     gc.collect()
    
#     # Training loop    
#     print(f"Starting training for {num_steps} steps...")
#     total_loss = 0.0
#     step_count = 0
    
#     # data_iter = dataloader(data, batch_size, key=data_rng)
    
    
#     # Set up tqdm progress bar
#     progress_bar = tqdm(range(num_steps), desc="Training", unit="step")
    
#     for step in progress_bar:
#         # Get next batch (already sharded across devices)
#         batch = next(data_iter)
        
#         # Perform distributed training step        
#         state, loss, train_rngs = train_step(state, batch, t1_replicated, train_rngs)
        
#         # Collect loss from all devices and average (move to CPU to avoid device sync issues)
#         loss = jax_utils.unreplicate(loss).item()
#         total_loss += loss
#         step_count += 1
        
#         # Update progress bar with current loss
#         progress_bar.set_postfix({"loss": f"{loss:.6f}", "avg_loss": f"{total_loss/step_count:.6f}"})
        
#         # Print detailed progress periodically
#         if (step+1) % print_every == 0 or step == num_steps - 1:
#             avg_loss = total_loss / step_count
#             print(f"\nStep {step+1}/{num_steps} | Avg Loss: {avg_loss}")
#             print_gpu_memory()
#             total_loss = 0.0
#             step_count = 0
            
#             jax.clear_caches()
#             gc.collect()
        
#         # Save checkpoint
#         if (step + 1) % checkpoint_every == 0 or step == num_steps - 1:
#             unreplicated_state = jax_utils.unreplicate(state)
#             checkpoints.save_checkpoint(
#                 ckpt_dir=checkpoint_dir,
#                 target=unreplicated_state,
#                 step=step + 1,
#                 overwrite=True,
#                 keep=3
#             )
#             print(f"Saved checkpoint at step {step+1}")
        
    
#     print("Training complete!")
    
#     # Generate samples
#     print("Generating samples...")
#     per_device_samples = (sample_size * sample_size) // num_devices
    
#     # Collect samples from all devices
#     sample_batches = []
    
#     for i in range(0, sample_size * sample_size, per_device_samples * num_devices):
#         # Force clear caches before generating samples
#         jax.clear_caches()
#         gc.collect()
        
#         # Split sample RNGs for each batch
#         batch_rngs = jr.split(sample_rngs[0], num_devices)
#         sample_rngs = jr.split(sample_rngs[1], num_devices)
        
#         # Generate samples on all devices
#         device_samples = generate_sample(
#             state, data_shape,dt0_replicated, t1_replicated, batch_rngs
#         )
        
#         # Move samples to CPU and flatten the device dimension
#         device_samples = np.array(device_samples)
#         sample_batches.append(device_samples)
        
#         # Update memory monitoring
#         if i % (2 * per_device_samples * num_devices) == 0:
#             print_gpu_memory()
    
#     # Combine all samples
#     samples = np.concatenate(sample_batches, axis=0)
#     samples = samples[:sample_size * sample_size]  # Ensure exact count
    
#     # Denormalize samples
#     samples = data_stats["mean"]+ data_stats["std"] * samples
#     samples = np.clip(samples, data_stats["min"], data_stats["max"])
    
#     # Arrange samples in a grid
#     sample_grid = einops.rearrange(
#         samples, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
#     )
    
#     # Plot and save samples
#     plt.figure(figsize=(10, 10))
#     plt.imshow(sample_grid, cmap="Greys")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig("diffusion_samples.png", dpi=300)
#     plt.show()
#     print("Samples saved to diffusion_samples.png")

# First, let's change the batch_loss_fn to avoid passing functions directly

def batch_loss_fn(model_apply, params, batch, t1, key):
    """Compute loss for a batch of data with internal noise schedule functions."""
    # Define noise schedule functions inside to avoid passing them as arguments
    int_beta = lambda t: t  # Linear schedule
    weight = lambda t: 1 - jnp.exp(-int_beta(t))  # Upweight region near t=0
    
    batch_size = batch.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    
    # Low-discrepancy sampling over t
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    
    # Vectorize the loss computation with integrated functions
    def single_loss_fn(data, t, key):
        """Compute loss for a single timestep with internal functions."""
        mean = data * jnp.exp(-0.5 * int_beta(t))
        var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
        std = jnp.sqrt(var)
        noise = jr.normal(key, data.shape)
        y = mean + std * noise
        
        # Apply model
        pred = model_apply({"params": params}, t, y)
        
        return weight(t) * jnp.mean((pred + noise / std) ** 2)
    
    # Vectorize the loss computation
    loss_fn = jax.vmap(single_loss_fn)
    
    return jnp.mean(loss_fn(batch, t, losskey))


# Modify the train_step function to avoid passing scalar values
@ft.partial(pmap, axis_name='devices')
def train_step(state, batch, t1_array, rng):
    """Distributed training step using pmap.
    
    Args:
        state: The replicated TrainState.
        batch: The data batch, already sharded across devices.
        t1_array: Array containing t1 value, replicated across devices.
        rng: Random key, replicated across devices.
    """
    # Extract t1 scalar from array
    t1 = t1_array[0]
    
    def loss_fn(params):
        loss = batch_loss_fn(state.apply_fn, params, batch, t1, rng)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Average gradients across all devices
    grads = pmean(grads, axis_name='devices')
    
    # Update parameters using optimizer
    state = state.apply_gradients(grads=grads)
    
    # Update random key
    new_rng = jr.fold_in(rng, 0)
    
    return state, loss, new_rng


# Similarly, let's modify the generate_sample function
@ft.partial(pmap, axis_name='devices')
def generate_sample(state, data_shape, dt0_array, t1_array, rng):
    """Generate sample on each device."""
    # Extract scalar values from arrays
    dt0 = dt0_array[0]
    t1 = t1_array[0]
    
    # Define int_beta inside to avoid passing as argument
    int_beta = lambda t: t  # Linear schedule
    
    def drift(t, y, args):
        # Calculate derivative of int_beta
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        model_output = state.apply_fn({"params": state.params}, t, y)
        return -0.5 * beta * (y + model_output)
    
    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(rng, data_shape)
    
    # Solve ODE from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]


# Now let's modify the main function to use these updated functions
def main(
    patch_size=4,
    hidden_size=64,
    mix_patch_size=512,
    mix_hidden_size=512,
    num_blocks=4,
    t1=10.0,
    num_steps=10_000,
    lr=3e-4,
    batch_size=256,
    print_every=1_000,
    checkpoint_every=5_000,
    dt0=0.1,
    sample_size=10,
    checkpoint_dir="./checkpoints",
    seed=5678,
    use_tqdm=True,  # Added option to toggle tqdm
):
    """Main training function with multi-GPU support."""
    # Check available devices
    num_devices = jax.local_device_count()
    percent_gpu = 0.4
    
#     os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
#     os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    print(f"Training with {num_devices} devices, using {percent_gpu}")
    
    # Adjust batch size to be divisible by number of devices
    if batch_size % num_devices != 0:
        orig_batch_size = batch_size
        batch_size = (batch_size // num_devices) * num_devices
        print(f"Adjusted batch size from {orig_batch_size} to {batch_size} to be divisible by {num_devices}")
    
    # Set up RNG keys - one per device
    master_rng = jr.PRNGKey(seed)
    model_rng, train_rng, data_rng, sample_rng = jr.split(master_rng, 4)
    
    # Create per-device RNG keys
    train_rngs = jr.split(train_rng, num_devices)
    sample_rngs = jr.split(sample_rng, num_devices)
    
    # Load and normalize data
    print("Loading MNIST dataset...")
    data = mnist()
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_max = jnp.max(data)
    data_min = jnp.min(data)
    data_shape = data.shape[1:]
    data = (data - data_mean) / data_std
    
    # Initialize model
    print("Initializing model...")
    model = Mixer2d(
        img_size=data_shape,
        patch_size=patch_size,
        hidden_size=hidden_size,
        mix_patch_size=mix_patch_size,
        mix_hidden_size=mix_hidden_size,
        num_blocks=num_blocks,
        t1=t1
    )
    
    # Create and distribute training state
    print("Creating training state...")
    state = create_train_state(model_rng, model, lr)
    
    # Replicate state across devices
    print("Replicating across gpus:")
    state = jax_utils.replicate(state)
    
    # Create arrays for scalar parameters and replicate across devices
    t1_array = jnp.array([t1])
    dt0_array = jnp.array([dt0])
    t1_replicated = jax_utils.replicate(t1_array)
    dt0_replicated = jax_utils.replicate(dt0_array)
    
    # Convert checkpoint_dir to absolute path and create directory
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    print(f"Using absolute checkpoint path: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {num_steps} steps...")
    total_loss = 0.0
    step_count = 0
    
    data_iter = dataloader(data, batch_size, key=data_rng)
    
    # Set up tqdm progress bar
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_steps), desc="Training", unit="step")
    
    for step in progress_bar:
        # Get next batch (already sharded across devices)
        batch = next(data_iter)
        
        # Perform distributed training step with properly shaped arguments
        state, loss, train_rngs = train_step(state, batch, t1_replicated, train_rngs)
        
        # Collect loss from all devices and average (move to CPU to avoid device sync issues)
        loss = jax_utils.unreplicate(loss).item()
        total_loss += loss
        step_count += 1
        
        # Update progress bar with current loss
        progress_bar.set_postfix({"loss": f"{loss:.6f}", "avg_loss": f"{total_loss/step_count:.6f}"})
        
        # Print detailed progress periodically
        if (step + 1) % print_every == 0 or step == num_steps - 1:
            avg_loss = total_loss / step_count
            print(f"\nStep {step+1}/{num_steps} | Avg Loss: {avg_loss:.6f}")
            total_loss = 0.0
            step_count = 0
        
        # Save checkpoint
        if (step + 1) % checkpoint_every == 0 or step == num_steps - 1:
            unreplicated_state = jax_utils.unreplicate(state)
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=unreplicated_state,
                step=step + 1,
                overwrite=True,
                keep=3
            )
            print(f"Saved checkpoint at step {step+1}")
    
    print("Training complete!")
    
    # Generate samples
    print("Generating samples...")
    per_device_samples = (sample_size * sample_size) // num_devices
    
    # Collect samples from all devices
    sample_batches = []
    
    # Set up tqdm for sample generation
    from tqdm.auto import tqdm
    total_batches = (sample_size * sample_size + per_device_samples * num_devices - 1) // (per_device_samples * num_devices)
    sample_progress = tqdm(range(total_batches), desc="Generating samples", unit="batch")
    
    for i in sample_progress:
        # Split sample RNGs for each batch
        batch_rngs = jr.split(sample_rngs[0], num_devices)
        sample_rngs = jr.split(sample_rngs[1], num_devices)
        
        # Generate samples on all devices with properly shaped arguments
        device_samples = generate_sample(
            state, data_shape, dt0_replicated, t1_replicated, batch_rngs
        )
        
        # Move samples to CPU and flatten the device dimension
        device_samples = np.array(device_samples)
        sample_batches.append(device_samples)
    
    # Combine all samples
    samples = np.concatenate(sample_batches, axis=0)
    samples = samples[:sample_size * sample_size]  # Ensure exact count
    
    # Denormalize samples
    samples = data_mean + data_std * samples
    samples = np.clip(samples, data_min, data_max)
    
    # Arrange samples in a grid
    sample_grid = einops.rearrange(
        samples, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    
    # Plot and save samples
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_grid, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("diffusion_samples.png", dpi=300)
    plt.show()
    print("Samples saved to diffusion_samples.png")


if __name__ == "__main__":
    main()