FILESEXTRAPATHS:prepend := "${THISDIR}/seatd:"

SRC_URI += "file://seatd.service"

do_install:append() {
    install -m 0644 ${UNPACKDIR}/seatd.service \
        ${D}${systemd_unitdir}/system/seatd.service
}
