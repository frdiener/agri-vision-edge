FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

# The data partition's ext4 fs is created on demand via x-systemd.makefs
RDEPENDS:${PN} += "e2fsprogs-mke2fs"

do_install:append() {
    install -d ${D}/data
}
