SUMMARY = "AVE base image"

LICENSE = "MIT"

inherit core-image

IMAGE_FEATURES += "ssh-server-openssh read-only-rootfs"

# 4096 byte block size to match dm-verity
EXTRA_IMAGECMD:ext4 += "-b 4096 -I 256 -O ^has_journal"

# Note: Make sure to generate your own salt!
VERITY_SALT = "8a8d8d807bd9838a80397a13b3bc13c55780ff1677ee4489366b17dab1b29316"

IMAGE_FSTYPES += "verity"
IMAGE_CLASSES += "image_types_verity"

IMAGE_FEATURES:append = " weston"

# Mesa
IMAGE_INSTALL:append = " \
    mesa-megadriver \
    libteflon \
    libegl-mesa \
    libgles2-mesa \
"

# OpenCV
IMAGE_INSTALL:append = " \
    opencv \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
"

# PACKAGECONFIG:append:pn-opencv = " \
#     gstreamer \
#     v4l \
#     gtk \
# "


# TFLite
IMAGE_INSTALL:append = " tensorflow-lite"

# https connections
IMAGE_INSTALL:append = " ca-certificates"

IMAGE_INSTALL:append = " python3-numpy tmux git rsync vim htop"
