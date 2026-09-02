require ptx-dev-keys.inc

SUMMARY = "Development keys for FIT image signing. Not to be used for production!"

PROVIDES = "virtual/fit-signing"

do_compile() {
    signing_import_define_role fit
    signing_import_cert_from_pem fit "${S}/fit/fit-4096-development.crt"
    signing_import_key_from_pem fit "${S}/fit/fit-4096-development.key"
}
