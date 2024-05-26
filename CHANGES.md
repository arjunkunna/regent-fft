# Version History

- 1.0: Initial Release (Apr 2024)
  - Supports CPU and single-GPU transforms for 1D, 2D, and 3D.
  - Supports Real-to-Complex and Complex-to-Complex transforms for both CPU
  - Supports single-precision (`float` / `complex32`) and double precision (`double` / `complex64`) transforms
  - Supports batched transforms: both R2C and C2C (complex32 and complex64)
    for GPUs, and R2C and C2C (only complex64) for CPUs.
