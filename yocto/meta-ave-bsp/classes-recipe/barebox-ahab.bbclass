# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025 Pengutronix, <yocto@pengutronix.de>
#
# Class for barebox AHAB signing
#
# Add "ahab" to MACHINE_FEATURES to make use of this class.

inherit signing barebox

BAREBOX_AHAB_SIGNING_KEY_ROLE[doc] = "The i.MX AHAB key role to use for signing barebox. Must be one of 'imx_ahab_srk{1,2,3,4}'"
BAREBOX_AHAB_SIGNING_KEY_ROLE ?= "imx_ahab_srk1"

BAREBOX_AHAB_SIGN_DIGEST[doc] = "Signature digest algorithm, must match the input given to cst for 'Enter the digest algorithm to use'."
BAREBOX_AHAB_SIGN_DIGEST ?= "sha384"

BAREBOX_AHAB_VERSION ?= "1.0"
BAREBOX_AHAB_REVOCATIONS ?= "0x0"

DEPENDS:append = "${@bb.utils.contains('MACHINE_FEATURES', 'ahab', \
                                       ' virtual/imx-ahab-signing extract-cert-native imx-cst-native', \
                                       '' , d)}"

python __anonymous() {
    # Without the ahab machine feature, this bbclass does nothing.
    if not bb.utils.contains("MACHINE_FEATURES", "ahab", True, False, d):
        return

    # Sanity checks
    key_role = d.getVar("BAREBOX_AHAB_SIGNING_KEY_ROLE")
    if key_role[:-1] != "imx_ahab_srk":
        bb.fatal("Unexpected AHAB signing key role")

    key_index = key_role[-1]
    if key_index not in "1234":
        bb.fatal("BAREBOX_AHAB_SIGNING_KEY_ROLE does not end with an allowed index")

    # Set which SRK the ROM code should validate against.
    d.setVar("SRK_INDEX", str(int(key_index) - 1))

    # Add SRK table generation and signing as prefuncs/postfuncs.
    d.appendVarFlag("do_configure", "prefuncs", " barebox_ahab_generate_imx_srk_table")
    d.appendVarFlag("do_compile", "postfuncs", " set_barebox_ahab_sign_images barebox_ahab_sign")
}

barebox_ahab_generate_imx_srk_table() {
        signing_prepare

        for i in 1 2 3 4; do
                signing_use_role "imx_ahab_srk${i}"
                extract-cert "${PKCS11_URI}" "${B}/srk${i}.der"
        done

        srktool --ahab_ver \
                --table "${B}/imx-srk-table.bin" \
                --efuses "${BAREBOX_ENV_DIR}/imx-srk-fuse.bin" \
                --digest sha256 \
                --sign_digest "${BAREBOX_AHAB_SIGN_DIGEST}" \
                --certs "${B}/srk1.der,${B}/srk2.der,${B}/srk3.der,${B}/srk4.der"
}
# Make sure no previous key material is accidentally used.
barebox_ahab_generate_imx_srk_table[cleandirs] = "${B}"

python set_barebox_ahab_sign_images() {
    # determine which barebox images to sign
    builddir = d.getVar("B")
    sign_images = []
    with open(os.path.join(builddir, "barebox-flash-images")) as images:
        for image in images:
            image = image.rstrip()
            with open(os.path.join(builddir, image), "rb") as f:
                f.seek(3)
                # If the header tag magic value is present, sign the image later.
                if f.read(1) == b"\x87":
                    sign_images.append(os.path.join(builddir, image))
                else:
                    bb.note(f"ignoring {image} due to unmatched i.MX93 magic")

    if not sign_images:
        bb.fatal("no images found to sign")

    d.setVar("BAREBOX_AHAB_SIGN_IMAGES", " ".join(sign_images))
}

barebox_ahab_sign() {
        signing_prepare
        signing_use_role "${BAREBOX_AHAB_SIGNING_KEY_ROLE}"

        for IMAGEFILE in ${BAREBOX_AHAB_SIGN_IMAGES}; do
                # Create the CSF for this particular image.
                CSF_FILE="$(basename ${IMAGEFILE})-csf"
                cat >${CSF_FILE} << EOF
[Header]
Target = AHAB
Version = ${BAREBOX_AHAB_VERSION}

[Install SRK]
File = "${B}/imx-srk-table.bin"
Source = "${PKCS11_URI}"
Source index = ${SRK_INDEX}
Source set = OEM
Revocations = ${BAREBOX_AHAB_REVOCATIONS}

[Authenticate Data]
File = "${IMAGEFILE}"
Offsets   = 0x0             0x90
EOF
                cst --input "${CSF_FILE}" --output "${IMAGEFILE}-signed" --backend pkcs11

        done

        # Now move the signed artifacts over the original images in a separatle loop, so
        # this hopefully does not accidentally re-run and sign images twice if something in
        # the previous loop failed.
        for IMAGEFILE in ${BAREBOX_AHAB_SIGN_IMAGES}; do
                mv "${IMAGEFILE}-signed" "${IMAGEFILE}"
        done
}
