COMPATIBLE_MACHINE:mx8mp = "^(mx8mp)$"
COMPATIBLE_MACHINE:mx93 = "^(mx93)$"
COMPATIBLE_MACHINE:k3 = "^(k3)"

TFA_PLATFORM:mx8mp = "imx8mp"
TFA_BUILD_TARGET:mx8mp = "bl31"

TFA_PLATFORM:mx93 = "imx93"
TFA_BUILD_TARGET:mx93 = "bl31"

TFA_BOARD:k3 = "lite"
TFA_PLATFORM:k3 = "k3"
TFA_BUILD_TARGET:k3 = "bl31"

# barebox decides to expect "firmware/imx8mp-bl31.bin" and tf-a recipe creates
# "firmware/trusted-firmware-a/bl31.bin".
do_install:append() {
    ln -sf ${PN}/$atfbin.bin ${D}/${FIRMWARE_BASE_DIR}/${TFA_PLATFORM}-$atfbin.bin
}

SYSROOT_DIRS += "${FIRMWARE_BASE_DIR}"

FILES:${PN} += "${FIRMWARE_BASE_DIR}"

EXTRA_OEMAKE:append = " IMX_BOOT_UART_BASE=auto"
