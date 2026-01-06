import os
import paddle

# This forces Python to look in your CUDA folder for the DLLs
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

# Now try the check
paddle.utils.run_check()