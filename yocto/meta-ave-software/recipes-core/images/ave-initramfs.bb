# SPDX-License-Identifier: MIT
#
# Copyright Pengutronix <yocto@pengutronix.de>
#

SUMMARY = "Initramfs image to boot a dm-verity based root file system"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

INITRAMFS_SCRIPTS = " \
    initramfs-framework-base \
    initramfs-module-debug \
    initramfs-module-udev \
    initramfs-module-verity \
"

# TODO: drop base-utils once initramfs-framework-base RDEPENDS on it
PACKAGE_INSTALL = "${INITRAMFS_SCRIPTS} ${VIRTUAL-RUNTIME_base-utils}"

# Don't allow the initramfs to contain a kernel
PACKAGE_EXCLUDE = "kernel-image-*"

# Do not pollute the initrd image with rootfs features
IMAGE_FEATURES = ""
IMAGE_LINGUAS = ""

# Do not add recommended or extra packages to the initrd image
NO_RECOMMENDATIONS = "1"
MACHINE_EXTRA_RDEPENDS = ""
MACHINE_EXTRA_RRECOMMENDS = ""

IMAGE_NAME_SUFFIX = ""
IMAGE_FSTYPES = "${INITRAMFS_FSTYPES}"

# don't make the image bigger than it is
IMAGE_ROOTFS_SIZE = "0"

inherit image image-artifact-names

IMAGE_PREPROCESS_COMMAND += "add_verity_params;"

add_verity_params () {
    cp -L "${DEPLOY_DIR_IMAGE}/ave-base-image${IMAGE_MACHINE_SUFFIX}.rootfs.ext4.verity-params" "${IMAGE_ROOTFS}/verity-params"
}

do_image[depends] += "ave-base-image:do_image_complete"
