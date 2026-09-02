LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

inherit genimage

COMPATIBLE_MACHINE = "(frdm-imx93)"

SRC_URI += "file://genimage.config"

DEPENDS += "e2fsprogs-native"

REFERENCE_ROOTFS_IMAGE = "ave-base-image"

GENIMAGE_VARIABLES[BAREBOX_BINARY] = "${BAREBOX_BINARY}"
GENIMAGE_VARIABLES[IMAGE_MACHINE_SUFFIX] = "${IMAGE_MACHINE_SUFFIX}"
GENIMAGE_VARIABLES[REFERENCE_ROOTFS_FULLNAME] = "${REFERENCE_ROOTFS_IMAGE}${IMAGE_MACHINE_SUFFIX}${IMAGE_NAME_SUFFIX}.ext4.verity"

do_genimage[depends] += " \
    virtual/bootloader:do_deploy \
    ave-fit-image:do_deploy \
    ${REFERENCE_ROOTFS_IMAGE}:do_image_complete \
"
