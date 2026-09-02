LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://COPYING.MIT;md5=838c366f69b72c5df05c96dff79b35f2"

inherit signing native

SRC_URI = "git://github.com/pengutronix/ptx-code-signing-dev.git;protocol=https;branch=master"
SRCREV = "391649c7c3559f2613f64d204c3ab4b9f407f67b"

B = "${WORKDIR}/build"

signing_import_cert_from_pem() {
    local role="${1}"
    local pem="${2}"

    signing_pkcs11_tool --type cert --write-object ${pem} --label "${role}"
}

signing_import_pubkey_from_pem() {
    local openssl_keyopt
    local role="${1}"
    local pem="${2}"

    if [ -n "${IMPORT_PASS_FILE}" ]; then
        openssl pkey \
            -passin "file:${IMPORT_PASS_FILE}" \
            -in "${pem}" -inform pem -pubout -outform der -out ${B}/pubkey_out.pem
    else
        openssl pkey \
            -in "${pem}" -inform pem -pubout -outform der -out ${B}/pubkey_out.pem
    fi
    signing_pkcs11_tool --type pubkey --write-object ${B}/pubkey_out.pem --label "${role}"
}

signing_import_privkey_from_pem() {
    local openssl_keyopt
    local role="${1}"
    local pem="${2}"

    if [ -n "${IMPORT_PASS_FILE}" ]; then
        openssl pkey \
            -passin "file:${IMPORT_PASS_FILE}" \
            -in "${pem}" -inform pem -outform der -out ${B}/privkey_out.pem
        signing_pkcs11_tool --type privkey --write-object ${B}/privkey_out.pem --label "${role}"
    else
        signing_pkcs11_tool --type privkey --write-object ${pem} --label "${role}"
    fi
}

do_compile:prepend() {
    signing_import_prepare
}

do_compile:append() {
    signing_import_finish
}

do_install() {
    signing_import_install
}

# Do not expose private keys in sstate artifacts
SSTATE_SKIP_CREATION:task-populate-sysroot = '1'
