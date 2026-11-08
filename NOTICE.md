# Third-party material

| Path | Origin | Licence | Use |
|---|---|---|---|
| `tools/nvclass/*.h` | NVIDIA [open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc) class headers (clc06f, clc0b5, clc1b5, clc1c0, QMD) | MIT (header text) | method/QMD layouts; `compute.py` parses `clc1c0qmd.h` at runtime |
| `tools/nvkm/` | Linux nouveau `drivers/gpu/drm/nouveau/nvkm` (subset, flat names) | MIT (file headers) | reference for every port in `pascal/` |
| `tools/mesa/nve4_compute.{c,h}` | Mesa `src/gallium/drivers/nouveau/nvc0` | MIT (file headers) | compute state + QMD values |
| `tools/envytools/` (not committed) | [envytools](https://github.com/envytools/envytools) | MIT | `envydis`/`envyas` (falcon + gm107 ISA), rnndb |
| firmware (not committed) | linux-firmware `nvidia/gp102` | LICENCE.nvidia | signed ACR/SEC2/GR/NVDEC blobs |
| runtime dependency | [tinygrad](https://github.com/tinygrad/tinygrad) 0.14.0 + TinyGPU DriverKit dext | MIT | PCIe transport (`APLRemotePCIDevice`) |
| `kernels/` toolchain | NVIDIA `nvidia-cuda-nvcc-cu12==12.4.131` (ptxas) in Docker | NVIDIA EULA | sm_61 cubins (built, not vendored) |

Prior art (not vendored, credited in README): `TheTom/pascal-egpu` (MIT), `Davinchy/macuda` (MIT).
