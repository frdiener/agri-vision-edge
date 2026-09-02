FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI += " \
    file://verityrootfs \
"

do_install:append() {
    install -m 0755 ${UNPACKDIR}/verityrootfs ${D}/init.d/05-verityrootfs
}

PACKAGES:append = " \
    initramfs-module-verity \
"

SUMMARY:initramfs-module-verity = "initramfs support for mounting a rootfs under dm-verity protection"
RDEPENDS:initramfs-module-verity = "${PN}-base libdevmapper util-linux-mountpoint"
FILES:initramfs-module-verity = "/init.d/05-verityrootfs"
