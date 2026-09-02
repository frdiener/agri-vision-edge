# Override the OE-Core mesa version with Mesa 26.2.1 and apply the
# Teflon/Etnaviv NPU patch series used for the AVE thesis experiments.
#
# Upstream base: Mesa 26.2.1 (commit 889476855143e855a7f92989251f09fb3b690cda)
# Patch source:  agri-vision-edge/patches/mesa  (branch: ave_master)
# SHA256 from:   Mesa release announcement (docs/relnotes/26.2.1.rst)

# Bump PV to 26.2.1
PV = "26.2.1"

# Replace the tarball checksum; clear the OE-Core 26.0.5 value first.
SRC_URI[sha256sum] = "c47e81bddc4760360a41ac3c5acec38acb81f9d750ecef47e7f3adc7021a4442"

# Append the Teflon / Etnaviv / Ethosu patch series.
# Patches are numbered 0001..0037 and apply cleanly on top of 26.2.1.
SRC_URI += " \
    file://0001-etnaviv-ml-use-a-single-arena-resource-instead-of-pe.patch \
    file://0002-etnaviv-ml-use-heap-allocations-during-compilation-d.patch \
    file://0003-etnaviv-ml-remove-screen-references-from-compilation.patch \
    file://0004-etnaviv-ml-match-output-dimension-swap-in-tiling-cal.patch \
    file://0005-etnaviv-ml-handle-NULL-bias-tensor-data-in-convoluti.patch \
    file://0006-etnaviv-ml-Use-NEON-for-signed-input-output-tensor-c.patch \
    file://0007-teflon-Support-standalone-ReLU6-activation.patch \
    file://0008-etnaviv-ml-Support-standalone-ReLU6-activation.patch \
    file://0009-teflon-Support-tensor-stacking-pack-operation.patch \
    file://0010-etnaviv-ml-fix-parameter-name-in-etna_ml_lower_logis.patch \
    file://0011-debug-teflon-Add-TEFLON_MAX_NODES-environment-variab.patch \
    file://0012-debug-teflon-Highlight-operation-support.patch \
    file://0013-debug-etnaviv-ml-Add-ETNA_NPU_MAX_CHANNELS-environme.patch \
    file://0014-etnaviv-ml-handle-depthwise-conv2d-with-single-chann.patch \
    file://0015-debug-etnaviv-ml-report-op-unsupport-reason.patch \
    file://0016-teflon-only-print-used-tensors.patch \
    file://0017-debug-etnaviv-ml-list-tensors-on-output-read.patch \
    file://0018-teflon-pass-output-tensor-indices-to-ml_subgraph_cre.patch \
    file://0019-etnaviv-ml-use-output-indices-in-lower_operations.patch \
    file://0020-debug-etnaviv-ml-debug-tensor-layout.patch \
    file://0021-etnaviv-ml-detranspose-3D-to-2D-reshapes-when-needed.patch \
    file://0022-debug-etnaviv-ml-log-reshape-detranspose-insertion.patch \
    file://0023-etnaviv-ml-treat-concatenation-operations-as-support.patch \
    file://0024-debug-teflon-allow-forcing-operations-unsupported-vi.patch \
    file://0025-debug-ml-log-operation-compatibility-checks.patch \
    file://0026-debug-teflon-etnaviv-improve-graph-partition-dumps.patch \
    file://0027-teflon-improve-ReLU6-range-checks-for-quantized-tens.patch \
    file://0028-etnaviv-ml-support-Leaky-ReLU-TP-operations.patch \
    file://0029-etnaviv-ml-fix-Leaky-ReLU-TP-scaling.patch \
    file://0030-ethosu-handle-standalone-PAD-operations.patch \
    file://0031-gallium-ml-propagate-nearest-resize-options.patch \
    file://0032-etnaviv-ml-support-2x-nearest-neighbor-resize.patch \
    file://0033-gallium-ml-add-pipe_tensor-rank.patch \
    file://0034-etnaviv-ml-don-t-transpose-concatenation-operands-th.patch \
    file://0035-etnaviv-ml-label-the-layout-a-3d-2d-RESHAPE-detransp.patch \
    file://0036-etnaviv-ml-reconcile-elementwise-operands-that-are-i.patch \
    file://0037-etnaviv-ml-bind-bypass-tensors-after-the-jobs-that-p.patch \
"

# Enable Teflon + Etnaviv/Ethosu NPU back-ends per SoC family
PACKAGECONFIG:append:mx8mp = " gallium etnaviv teflon"
PACKAGECONFIG:append:mx93  = " gallium ethosu teflon"
