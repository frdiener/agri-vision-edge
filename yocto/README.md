# YOCTO.BSP-AVE

Yocto BSP workspace for the embedded software images evaluated in the
bachelor thesis on NPU-accelerated inference on NXP i.MX SoCs.

## Purpose

Build the embedded Linux images used for the NPU inference measurements in the
thesis.  Targets are the **NXP i.MX8M Plus FRDM** (Etnaviv/NPU) and
**NXP i.MX93 FRDM** (Ethosu/NPU) boards running the custom `ave` distro.

## Project layers

| Layer | Purpose |
|-------|---------|
| `meta-ave-bsp` | BSP layer: machine definitions, kernel, barebox, NXP firmware, FIT-image signing, hardware integration for frdm-imx8mp and frdm-imx93 |
| `meta-ave-software` | Software layer: `ave` distro config, root-filesystem image, Mesa/Teflon/NPU patches, Weston compositor config, TFLite integration |

External layers (`bitbake`, `openembedded-core`, `meta-openembedded`,
`meta-arm`, `meta-tensorflow`, `meta-ptx`) are **not vendored** in this
repository.  They are checked out at exact pinned commits by `setup.sh`.

## Setup

Clone this repository and run the setup script:

```bash
git clone <repo-url> YOCTO.BSP-AVE
cd YOCTO.BSP-AVE
./setup.sh
```

`setup.sh` reads `sources.lock` and clones each external layer at the recorded
SHA in detached-HEAD state.  It is idempotent and safe to re-run.

After setup, initialise the build environment:

```bash
. ./oe-init-build-env [build-dir]   # defaults to build/
```

The first time you source `oe-init-build-env`, `bblayers.conf` and
`local.conf` are generated from the templates in
`meta-ave-bsp/conf/templates/default/`.

## Builds

### i.MX8M Plus FRDM (Etnaviv NPU)

```bash
MACHINE=frdm-imx8mp bitbake ave-base-image
# Full disk image:
MACHINE=frdm-imx8mp bitbake fdi-frdm-imx8mp-disk-image
```

### i.MX93 FRDM (Ethosu NPU)

```bash
MACHINE=frdm-imx93 bitbake ave-base-image
# Full disk image:
MACHINE=frdm-imx93 bitbake fdi-frdm-imx93-disk-image
```

The default machine (set in `local.conf.sample`) is `frdm-imx93`.

## Dependency pinning

External Yocto layers are not vendored.  `sources.lock` records the canonical
upstream URL and exact Git SHA for each dependency:

| Component | URL | SHA |
|-----------|-----|-----|
| `bitbake` | <https://git.openembedded.org/bitbake> | `cb28befc56d2` |
| `openembedded-core` | <https://git.openembedded.org/openembedded-core> | `cc9037a4c44e` |
| `meta-openembedded` | <https://github.com/openembedded/meta-openembedded> | `420222862f5a` |
| `meta-arm` | <https://git.yoctoproject.org/meta-arm> | `eb9b2afff9e2` |
| `meta-tensorflow` | <https://git.yoctoproject.org/meta-tensorflow> | `a5cff71e96f3` |
| `meta-ptx` | <https://github.com/pengutronix/meta-ptx> | `34d31c886545` |

Full SHAs and provenance notes are in `sources.lock`.

## Mesa / Teflon NPU patch series

`meta-ave-software/recipes-graphics/mesa/` overrides the OE-Core mesa recipe
to build **Mesa 26.2.1** instead of the default version and applies 37 patches
on top of it.

The patches implement or improve Teflon/Etnaviv/Ethosu NPU operator support:
tensor-layout fixes, ReLU6, tensor stacking (pack), Leaky ReLU, resize,
concatenation, bypass-tensor sequencing, and various delegation diagnostics.

```
upstream base: Mesa 26.2.1 (commit 889476855143e855a7f92989251f09fb3b690cda)
```

## Build outputs

Yocto places deployment images under:

```
<build-dir>/tmp/deploy/images/<machine>/
```
