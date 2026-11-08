; Read an MMIO register from the falcon's side, exactly the way the ACR app does.
;
; The app's read32 (imem 0x27fc) pokes SEC2's external-port registers at
; I[0x1c000..0x1c300] (= host MMIO 0x700..0x70c):
;
;     I[0x1c300] = 0x20000            setup
;     wait for I[0x1c000] bits[14:12] == 0
;     I[0x1c100] = address
;     I[0x1c000] = 0x800000f1         read command (data appears in I[0x1c200])
;     wait again
;
; The host cannot use that port (BAR0 reads there return the 0xdead5ec1
; access-denied sentinel), so only falcon code can read the registers the ACR app
; is judging.  Results are written to DMEM where the host can PIO them back:
;
;     D[0x1000] value read          D[0x1004] 0x600d0000 ready / 0x7ead0000 timeout
;     D[0x1008] command register    D[0x100c] I[0x9100] (host 0x244, the status the
;                                             app polls before every access)
;
; The address to read is patched in by the loader (PEEK_ADDR).

mov $r15 0x1000
mov $r11 0x20000
mov $r10 0x1c300
iowrs I[$r10] $r11
mov $r9 0x1000
mov $r11 {ADDR}
mov $r10 0x1c100
iowrs I[$r10] $r11
mov $r11 0x800000f1
mov $r10 0x1c000
iowrs I[$r10] $r11
wait:
mov $r10 0x1c000
iord $r1 I[$r10]
extr $r1 $r1 0xc:0xe
bra e ready
sub b32 $r9 $r9 0x1
bra e timeout
lbra wait
ready:
mov $r10 0x1c200
iord $r0 I[$r10]
st b32 D[$r15] $r0
mov $r0 0x600d0000
st b32 D[$r15+0x4] $r0
lbra out
timeout:
mov $r0 0x7ead0000
st b32 D[$r15+0x4] $r0
out:
mov $r10 0x1c000
iord $r0 I[$r10]
st b32 D[$r15+0x8] $r0
mov $r10 0x9100
iord $r0 I[$r10]
st b32 D[$r15+0xc] $r0
exit
