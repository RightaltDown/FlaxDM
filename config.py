import os

def setup_jax_environment(gpu_ids='0,1,3', mem_fraction='0.8', allocator='platform'):
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = mem_fraction
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = allocator
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids