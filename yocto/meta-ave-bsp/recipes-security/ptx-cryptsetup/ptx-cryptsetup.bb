SUMMARY = "Create and manage an encrypted block device"
DESCRIPTION = "Script and systemd unit for setting up a virtual block \
device via dm-crypt utilizing trusted keys"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://ptx-cryptsetup \
    file://ptx-cryptsetup@.service \
    file://40-device-timeout.conf \
"

S = "${UNPACKDIR}"

RDEPENDS:${PN} = " \
    busybox \
    diffutils \
    keyutils \
    libdevmapper \
    lvm2-udevrules \
    util-linux-blockdev \
    util-linux-wipefs \
"

# Use allarch to indicate that the package is architecture-independent and
# ensure noarch packaging
inherit allarch systemd
PACKAGE_ARCH = "all"

VOLUMES[doc] = "Space-separated list of partition labels matching the mapper device names"
VOLUMES ?= "data"

do_install() {
    install -d ${D}${sbindir}
    install -m 755 ${UNPACKDIR}/ptx-cryptsetup ${D}${sbindir}/

    install -d ${D}${systemd_system_unitdir}

    install -m 644 ${UNPACKDIR}/ptx-cryptsetup@.service ${D}${systemd_system_unitdir}

    for ptx_cryptsetup_volume in ${VOLUMES}; do
        # add device timeout drop-in for mapper device
        install -d ${D}${systemd_system_unitdir}/dev-mapper-${ptx_cryptsetup_volume}.device.d
        install -m 644 ${UNPACKDIR}/40-device-timeout.conf ${D}${systemd_system_unitdir}/dev-mapper-${ptx_cryptsetup_volume}.device.d
    done
}

SYSTEMD_SERVICE:${PN} = "${@' '.join(f'ptx-cryptsetup@{vol}.service' for vol in d.getVar('VOLUMES').split())}"

FILES:${PN}:append = " ${systemd_system_unitdir}"
