# SPDX-License-Identifier: MIT
#
# Copyright (C) 2024 Pengutronix, <yocto@pengutronix.de>
#
# Class for barebox HABv4 signing
#
# Add "habv4" to MACHINE_FEATURES to make use of this class.

inherit signing

BAREBOX_HABV4_SIGNING_KEY_ROLE[doc] = "The i.MX HABv4 key role to use for signing barebox. Must be one of 'imx_habv4_srk{1,2,3,4}'"
BAREBOX_HABV4_SIGNING_KEY_ROLE ?= "imx_habv4_srk1"

DEPENDS:append = "${@bb.utils.contains('MACHINE_FEATURES', 'habv4', \
                                       ' virtual/imx-hab-signing extract-cert-native imx-cst-native', \
                                       '' , d)}"

python __anonymous() {
    if not bb.utils.contains("MACHINE_FEATURES", "habv4", True, False, d):
        return

    key_role = d.getVar("BAREBOX_HABV4_SIGNING_KEY_ROLE")
    if key_role[:-1] != "imx_habv4_srk" or key_role[-1] not in "1234":
        bb.fatal(f"Unexpected BAREBOX_HABV4_SIGNING_KEY_ROLE, expected 'imx_habv4_srk{1,2,3,4}', got '{key_role}'")

    key_index = key_role[-1]

    # Counter starts at 1.
    d.setVar("BAREBOX_SRK_INDEX", key_index)

    # Add SRK table generation and signing preparation as prefuncs/postfuncs.
    d.appendVarFlag("do_configure", "prefuncs", " barebox_habv4_generate_imx_srk_table")
    d.appendVarFlag("do_compile", "prefuncs", " barebox_habv4_srk_index_check")
}

python barebox_habv4_srk_index_check() {
    # Retrieve CONFIG_HABV4_SRK_INDEX from barebox config. Counter starts at 0.
    config_key_index, _ = bb.process.run(f"{d.getVar('S')}/scripts/config --file {d.getVar('B')}/.config --state HABV4_SRK_INDEX")
    config_key_index = config_key_index.strip()

    expected_key_index = str(int(d.getVar("BAREBOX_SRK_INDEX")) - 1)
    if config_key_index != expected_key_index:
        bb.fatal(f"Unexpected CONFIG_HABV4_SRK_INDEX, expected '{expected_key_index}', got '{config_key_index}'")
}

barebox_habv4_generate_imx_srk_table() {
    signing_prepare

    for i in 1 2 3 4; do
        signing_use_role "imx_habv4_srk${i}"
        extract-cert "${PKCS11_URI}" "${B}/srk${i}.der"
    done

    srktool --hab_ver 4 \
            --table "${B}/imx-srk-table.bin" \
            --efuses "${BAREBOX_ENV_DIR}/imx-srk-fuse.bin" \
            --digest sha256 \
            --certs "${B}/srk1.der,${B}/srk2.der,${B}/srk3.der,${B}/srk4.der"

}
# Make sure no previous key material is accidentally used.
barebox_habv4_generate_imx_srk_table[cleandirs] = "${B}"

do_compile:prepend() {
    if ${@bb.utils.contains('MACHINE_FEATURES', 'habv4', 'true', 'false', d)}; then
        signing_prepare

        export CONFIG_HABV4_TABLE_BIN="${B}/imx-srk-table.bin"

        # Make the imx-cst backend use the PKCS#11 API.
        export CST_EXTRA_CMDLINE_OPTIONS="-b pkcs11"

        # Pass the PKCS11 URIs for the HABv4 CSF/IMG signing, key role indexes
        # derived from BAREBOX_SRK_INDEX.
        signing_use_role "imx_habv4_csf${BAREBOX_SRK_INDEX}"
        export CONFIG_HABV4_CSF_CRT_PEM="${PKCS11_URI}"

        signing_use_role "imx_habv4_img${BAREBOX_SRK_INDEX}"
        export CONFIG_HABV4_IMG_CRT_PEM="${PKCS11_URI}"
    fi
}
