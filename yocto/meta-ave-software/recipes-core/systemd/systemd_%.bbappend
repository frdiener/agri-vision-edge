FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI += "\
    file://00-resolved-run-mount.conf \
    file://00-networkd-persist-var-lib-mount.conf \
    file://00-timesyncd-var-lib-mount.conf \
"

PACKAGECONFIG:remove = " \
    vconsole \
"

PACKAGECONFIG:append = " \
    journal-color \
"

do_install:append() {
    install -d ${D}${systemd_system_unitdir}/systemd-resolved.service.d/
    install -m 0644 ${UNPACKDIR}/00-resolved-run-mount.conf  ${D}${systemd_system_unitdir}/systemd-resolved.service.d/

    install -d ${D}${systemd_system_unitdir}/systemd-networkd-persistent-storage.service.d/
    install -m 0644 ${UNPACKDIR}/00-networkd-persist-var-lib-mount.conf  ${D}${systemd_system_unitdir}/systemd-networkd-persistent-storage.service.d/

    install -d ${D}${systemd_system_unitdir}/systemd-timesyncd.service.d/
    install -m 0644 ${UNPACKDIR}/00-timesyncd-var-lib-mount.conf  ${D}${systemd_system_unitdir}/systemd-timesyncd.service.d/
}
