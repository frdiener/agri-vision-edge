SECTION = "bootloaders"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=f5125d13e000b9ca1f0d3364286c4192"

inherit barebox barebox-habv4 barebox-ahab signing python3native

DEPENDS += "lzop-native lz4-native virtual/fit-signing"
DEPENDS:append:mx8mp = " firmware-imx trusted-firmware-a"
DEPENDS:append:mx93 = " firmware-imx trusted-firmware-a"
DEPENDS:append:beagleplay-k3r5 = " ti-sci-fw python3-yamllint-native python3-jsonschema-native"

# define as variable to make it easy to override
SRC_URI_MAIN ?= "https://barebox.org/download/barebox-${PV}.tar.bz2"

SRC_URI = "\
    ${SRC_URI_MAIN} \
    file://defconfig \
    ${SRC_URI_BOOTCHOOSER} \
"

SRC_URI_BOOTCHOOSER = " \
    file://dts/bootstate.dtsi \
    file://dts/env.dtsi \
    file://env/nv/boot.default \
    file://env/nv/bootchooser.state_prefix \
    file://env/nv/bootchooser.targets \
    file://env/boot/system0 \
    file://env/boot/system1 \
"
SRC_URI_BOOTCHOOSER:beagleplay-k3r5 = ""

SRC_URI:append:tq-mba8mpxl = " file://env/nv/dev.mmc1.broken_cd"
SRC_URI:append:riotboard = " file://env/nv/dev.mmc2.probe"

BAREBOX_FIRMWARE_DIR = "${STAGING_DIR_TARGET}/firmware"

# Let barebox integrate FIT image keys via CONFIG_CRYPTO_PUBLIC_KEYS using the
# FIT_KEY env variable holding the PKCS#11 URI.
do_compile:prepend() {
    signing_prepare
    signing_use_role fit
    export FIT_KEY="${PKCS11_URI}"
}

COMPATIBLE_MACHINE = "^$"
COMPATIBLE_MACHINE:frdm-imx93 = "^(frdm-imx93)$"
COMPATIBLE_MACHINE:frdm-imx8mp = "^(frdm-imx8mp)$"
