require ptx-dev-keys.inc

SUMMARY = "Development keys for RAUC bundle signing. Not to be used for production!"

PROVIDES = "virtual/rauc-signing"

do_compile() {
    signing_import_define_role rauc
    signing_import_cert_from_pem rauc "${S}/rauc/rauc.cert.pem"
    signing_import_key_from_pem rauc "${S}/rauc/rauc.key.pem"
}
