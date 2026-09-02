SUMMARY = "Freescale i.MX8 firmware files"

FSL_MIRROR = "https://www.nxp.com/lgfiles/NMG/MAD/YOCTO"

require firmware-imx-${PV}.inc

PACKAGE_ARCH = "${MACHINE_ARCH}"

do_install() {
    install -d ${D}/firmware

    # Synopsys DDR
    for ddr_firmware in ${DDR_FIRMWARE_FILES}; do
        install -m 0644 ${S}/firmware/ddr/synopsys/${ddr_firmware} ${D}/firmware
    done
}

FILES:${PN} = "/firmware"
SYSROOT_DIRS += "/firmware"
FILES:${PN}-dbg = "/firmware/*.elf"
