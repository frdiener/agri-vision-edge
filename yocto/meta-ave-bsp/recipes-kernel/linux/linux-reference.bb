inherit kernel

SECTION = "kernel"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"

PV = "7.1.4"

SRC_URI = "https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-${PV}.tar.xz"
SRC_URI += "file://defconfig"

SRC_URI[sha256sum] = "1c63922a119675d38e3ae0f8f6ee07f15c41a786ab9ed66563749bb8c9a08e2e"

S = "${UNPACKDIR}/linux-${PV}"

DEPENDS += "lzop-native"

SRC_URI += "file://0001-ARM-Don-t-mention-the-full-path-of-the-source-direct.patch"
SRC_URI += "file://0001-arm64-dts-imx8mp-frdm-Add-missing-HDMI-DDC-pinctrl.patch"

SRC_URI += " file://imx93-11x11-frdm.dts"

COMPATIBLE_MACHINE = ""
