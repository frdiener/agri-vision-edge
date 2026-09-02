SUMMARY = "AVE FIT image to boot a dm-verity based root file system"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit fitimage signing

DEPENDS:append = " virtual/fit-signing"

FITIMAGE_SIGNING_KEY_ROLE ?= "fit"

FITIMAGE_IMAGES = "kernel fdt ramdisk"

FITIMAGE_IMAGE_kernel = "virtual/kernel"
FITIMAGE_IMAGE_kernel[type] = "kernel"

KERNEL_COMPRESSION = "none"
KERNEL_COMPRESSION:frdm-imx8mp = "gzip"
KERNEL_COMPRESSION:frdm-imx93 = "gzip"
FITIMAGE_IMAGE_kernel[comp] = "${KERNEL_COMPRESSION}"

FITIMAGE_IMAGE_fdt = "virtual/kernel"
FITIMAGE_IMAGE_fdt[type] = "fdt"

FITIMAGE_IMAGE_ramdisk = "ave-initramfs"
FITIMAGE_IMAGE_ramdisk[type] = "ramdisk"

FITIMAGE_SIGN = "1"
FITIMAGE_MKIMAGE_EXTRA_ARGS = "--engine pkcs11"
FITIMAGE_SIGN_KEYDIR = "${PKCS11_URI#pkcs11:}"

do_fitimage:prepend() {
    signing_prepare
    signing_use_role "${FITIMAGE_SIGNING_KEY_ROLE}"
}
