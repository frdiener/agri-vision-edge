#!/usr/bin/env python3
"""Compact, robust-ish CSV logger for an FNIRSI FNB58.

Output columns:
    t_ns,voltage_V,current_A,power_W

`t_ns` is a host monotonic timestamp in nanoseconds. The FNB58 sends groups of
four values without per-value timestamps; values within a report are assigned
nominal 10 ms spacing (100 Hz), with the newest value aligned to USB receipt.

Use a .gz output filename to write gzip-compressed CSV directly.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path
from typing import TextIO

import usb.core  # pyright: ignore[reportMissingImports]
import usb.util  # pyright: ignore[reportMissingImports]

VID = 0x2E3C
PID = 0x5558
REPORT_SIZE = 64
SAMPLES_PER_REPORT = 4
SAMPLE_PERIOD_NS = 10_000_000  # FNB58 protocol is nominally 100 Hz.


def make_command(code: int, crc: int) -> bytes:
    return b"\xaa" + bytes([code]) + b"\x00" * 61 + bytes([crc])


# The startup sequence begins a streaming window. Command 0x83 extends that
# window at low cadence; sending it for every report eventually wedges the meter.
START_COMMANDS = (
    make_command(0x81, 0x8E),
    make_command(0x82, 0x96),
    make_command(0x82, 0x96),
)
CONTINUE_COMMAND = make_command(0x83, 0x9E)


def crc8(payload: bytes) -> int:
    """CRC-8: poly 0x39, init 0x42, no reflection, xorout 0."""
    value = 0x42
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (
                ((value << 1) ^ 0x39) & 0xFF if value & 0x80 else (value << 1) & 0xFF
            )
    return value


def u32le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def decode(report: bytes) -> list[tuple[float, float]] | None:
    """Return [(voltage_V, current_A), ...], or None for non-data reports."""
    if len(report) != REPORT_SIZE:
        raise ValueError(f"expected {REPORT_SIZE} bytes, got {len(report)}")
    if report[0] != 0xAA or report[1] != 0x04:
        return None
    if report[-1] != crc8(report[1:-1]):
        raise ValueError("CRC mismatch")

    result: list[tuple[float, float]] = []
    for index in range(SAMPLES_PER_REPORT):
        offset = 2 + 15 * index
        voltage = u32le(report, offset) / 100_000.0
        current = u32le(report, offset + 4) / 100_000.0
        result.append((voltage, current))
    return result


def open_meter():
    device = usb.core.find(idVendor=VID, idProduct=PID)
    if device is None:
        raise RuntimeError("FNB58 not found (expected USB VID:PID 2e3c:5558)")

    # Reset the retained firmware session. This meter can re-enumerate during
    # reset, so the original PyUSB object must never be used afterwards.
    try:
        device.reset()
    except usb.core.USBError as exc:
        if getattr(exc, "errno", None) != 2:  # ENOENT can mean re-enumeration.
            raise RuntimeError(f"could not reset FNB58 USB session: {exc}") from exc
    finally:
        usb.util.dispose_resources(device)

    time.sleep(0.25)
    deadline = time.monotonic() + 5.0
    device = None
    while device is None and time.monotonic() < deadline:
        device = usb.core.find(idVendor=VID, idProduct=PID)
        if device is None:
            time.sleep(0.1)
    if device is None:
        raise RuntimeError("FNB58 did not re-enumerate within 5 seconds after reset")

    interface_number = None
    for config in device:
        for interface in config:
            if interface.bInterfaceClass == 0x03:  # HID
                interface_number = interface.bInterfaceNumber
                break
        if interface_number is not None:
            break
    if interface_number is None:
        usb.util.dispose_resources(device)
        raise RuntimeError("FNB58 HID interface not found")

    try:
        if device.is_kernel_driver_active(interface_number):
            device.detach_kernel_driver(interface_number)
    except NotImplementedError:
        pass
    except usb.core.USBError as exc:
        usb.util.dispose_resources(device)
        raise RuntimeError(f"could not detach FNB58 HID driver: {exc}") from exc

    try:
        # This may return BUSY if the existing configuration is already active.
        try:
            device.set_configuration()
        except usb.core.USBError as exc:
            if getattr(exc, "errno", None) != 16:
                raise

        interface = device.get_active_configuration()[(interface_number, 0)]
        endpoint_in = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: (
                usb.util.endpoint_direction(endpoint.bEndpointAddress)
                == usb.util.ENDPOINT_IN
            ),
        )
        endpoint_out = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: (
                usb.util.endpoint_direction(endpoint.bEndpointAddress)
                == usb.util.ENDPOINT_OUT
            ),
        )
        if endpoint_in is None or endpoint_out is None:
            raise RuntimeError("could not find FNB58 HID IN/OUT endpoints")

        usb.util.claim_interface(device, interface_number)
    except BaseException:
        usb.util.dispose_resources(device)
        raise

    # Do not reattach hid-generic on close. Rebinding it can hang this meter and
    # remove it from USB enumeration; a later invocation can claim it directly.
    return device, interface_number, endpoint_in, endpoint_out


def open_csv(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdout, False

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".gz":
        return gzip.open(destination, "wt", encoding="utf-8", newline=""), True
    return destination.open("w", encoding="utf-8", newline=""), True


def write_command(endpoint_out, command: bytes, description: str) -> None:
    """Write one protocol command and turn opaque PyUSB errors into context."""
    try:
        endpoint_out.write(command, timeout=1_000)
    except usb.core.USBError as exc:
        raise RuntimeError(f"FNB58 {description} failed: {exc}") from exc


def drain_reports(endpoint_in, timeout_ms: int, max_seconds: float = 3.0) -> int:
    """Consume queued reports until quiet, but never block cleanup forever."""
    drained = 0
    deadline = time.monotonic() + max_seconds
    while True:
        remaining_ms = int((deadline - time.monotonic()) * 1_000)
        if remaining_ms <= 0:
            return drained
        read_timeout_ms = max(1, min(timeout_ms, remaining_ms))
        try:
            endpoint_in.read(REPORT_SIZE, timeout=read_timeout_ms)
            drained += 1
        except usb.core.USBTimeoutError:
            return drained
        except usb.core.USBError:
            # Cleanup must remain best-effort if the meter was disconnected.
            return drained
        except KeyboardInterrupt:
            # A second Ctrl-C must not skip interface cleanup and recreate the
            # stale-report problem this drain is intended to prevent.
            continue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact FNB58 voltage/current/power logger"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="CSV path; use .gz for gzip compression, '-' for stdout",
    )
    parser.add_argument(
        "--flush-seconds",
        type=float,
        default=1.0,
        help="flush interval; 0 flushes every report",
    )
    args = parser.parse_args()
    if args.flush_seconds < 0:
        parser.error("--flush-seconds must be >= 0")

    stream, close_stream = open_csv(args.output)
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("t_ns", "voltage_V", "current_A", "power_W"))
    stream.flush()

    device = None
    interface_number = None
    endpoint_in = None
    valid_reports = invalid_reports = 0

    try:
        device, interface_number, endpoint_in, endpoint_out = open_meter()

        for index, command in enumerate(START_COMMANDS, start=1):
            write_command(endpoint_out, command, f"startup command {index}")
        time.sleep(0.1)

        next_continue_ns = time.monotonic_ns() + 1_000_000_000
        next_flush_ns = time.monotonic_ns() + int(args.flush_seconds * 1_000_000_000)
        print(
            "Logging FNB58 at nominal 100 Hz; continuation request every 1 s. "
            "Press Ctrl-C to stop.",
            file=sys.stderr,
        )

        while True:
            try:
                report = bytes(endpoint_in.read(REPORT_SIZE, timeout=1_000))
                received_ns = time.monotonic_ns()
            except usb.core.USBTimeoutError as exc:
                # Do not issue another request: the timed-out one may still be
                # pending, and a second OUT write can wedge the meter.
                raise RuntimeError("FNB58 report read timed out") from exc

            try:
                samples = decode(report)
            except ValueError as exc:
                invalid_reports += 1
                print(f"warning: skipped report ({exc})", file=sys.stderr)
                samples = None

            if samples is not None:
                for index, (voltage, current) in enumerate(samples):
                    t_ns = (
                        received_ns
                        - (SAMPLES_PER_REPORT - 1 - index) * SAMPLE_PERIOD_NS
                    )
                    writer.writerow(
                        (
                            t_ns,
                            f"{voltage:.6f}",
                            f"{current:.6f}",
                            f"{voltage * current:.6f}",
                        )
                    )
                valid_reports += 1

            now_ns = time.monotonic_ns()
            if now_ns >= next_continue_ns:
                write_command(endpoint_out, CONTINUE_COMMAND, "continuation request")
                next_continue_ns = now_ns + 1_000_000_000

            if args.flush_seconds == 0 or now_ns >= next_flush_ns:
                stream.flush()
                next_flush_ns = now_ns + int(args.flush_seconds * 1_000_000_000)

    except KeyboardInterrupt:
        print(
            f"Stopped: {valid_reports} valid, {invalid_reports} invalid reports.",
            file=sys.stderr,
        )
        return 0
    except (RuntimeError, usb.core.USBError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if endpoint_in is not None:
            # Let the last continuation window run dry before releasing the
            # interface. Bound the drain in case broken firmware never goes quiet.
            drain_reports(endpoint_in, timeout_ms=1_000, max_seconds=3.0)
        if device is not None and interface_number is not None:
            try:
                usb.util.release_interface(device, interface_number)
            except usb.core.USBError:
                pass
            usb.util.dispose_resources(device)
        if close_stream:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
