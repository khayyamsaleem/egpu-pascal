# firmware/ (not in git)

The bring-up needs NVIDIA's **signed** GP102 firmware (ACR, SEC2 RTOS, FECS/GPCCS, the
NVDEC VPR scrubber) and **this card's VBIOS**. Neither is committed:

* signed blobs: `./fetch.sh` pulls `nvidia/gp102/{acr,gr,sec2,nvdec}` from
  [linux-firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git)
  (redistributable under its `LICENCE.nvidia`);
* VBIOS: `~/.venvs/tinygpu/bin/python pascal/dump_vbios.py` reads it from the card
  (BAR0 + 0x300000) into `gp102_vbios.rom`.

`SHA256SUMS` lists the exact files this project was validated with (EVGA GTX 1080 Ti,
subsystem `3842:6696`; the VBIOS hash will differ on other boards).
