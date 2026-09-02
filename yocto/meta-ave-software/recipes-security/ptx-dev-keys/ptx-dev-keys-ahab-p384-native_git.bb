require ptx-dev-keys.inc

SUMMARY = "Development keys for i.MX AHAB signing. Not to be used for production!"

PROVIDES = "virtual/imx-ahab-signing"

do_compile() {
    export IMPORT_PASS_FILE="${S}/ahab/p384r1/keys/key_pass.txt"

    for i in 1 2 3 4; do
        r="imx_ahab_srk${i}"
        signing_import_define_role ${r}
        signing_import_cert_from_der ${r} ${S}/ahab/p384r1/crts/SRK${i}_sha384_secp384r1_v3_usr_crt.der
        signing_import_key_from_pem ${r} ${S}/ahab/p384r1/keys/SRK${i}_sha384_secp384r1_v3_usr_key.pem
    done
}
