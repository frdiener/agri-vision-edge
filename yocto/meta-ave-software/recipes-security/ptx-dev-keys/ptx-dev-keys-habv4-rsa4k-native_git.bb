require ptx-dev-keys.inc

SUMMARY = "Development keys for i.MX HABv4 signing. Not to be used for production!"

PROVIDES = "virtual/imx-hab-signing"

do_compile() {
    export IMPORT_PASS_FILE="${S}/habv4/keys/key_pass.txt"

    for i in 1 2 3 4; do
        r="imx_habv4_srk${i}"
        signing_import_define_role ${r}
        signing_import_cert_from_der ${r} ${S}/habv4/crts/SRK${i}_sha256_4096_65537_v3_ca_crt.der
        signing_import_key_from_pem ${r} ${S}/habv4/keys/SRK${i}_sha256_4096_65537_v3_ca_key.pem

        r="imx_habv4_csf${i}"
        signing_import_define_role ${r}
        signing_import_cert_from_der ${r} ${S}/habv4/crts/CSF${i}_1_sha256_4096_65537_v3_usr_crt.der
        signing_import_key_from_pem ${r} ${S}/habv4/keys/CSF${i}_1_sha256_4096_65537_v3_usr_key.pem

        r="imx_habv4_img${i}"
        signing_import_define_role ${r}
        signing_import_cert_from_der ${r} ${S}/habv4/crts/IMG${i}_1_sha256_4096_65537_v3_usr_crt.der
        signing_import_key_from_pem ${r} ${S}/habv4/keys/IMG${i}_1_sha256_4096_65537_v3_usr_key.pem
    done
}
