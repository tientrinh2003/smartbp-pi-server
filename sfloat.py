#!/usr/bin/env python3
"""Utility for decoding IEEE-11073 16-bit SFLOAT values and parsing a BLE blood pressure packet.

Provides:
- _sfloat_to_float(raw_uint16)
- parse_measurement(data: bytearray)

Includes a small CLI test that decodes example bytes.
"""
from typing import Optional


def _sfloat_to_float(raw_uint16: int) -> float:
    """Convert IEEE-11073 16-bit SFLOAT to Python float.

    SFLOAT: 16-bit value where bits 0-11 are a 12-bit signed mantissa and bits 12-15
    are a 4-bit signed exponent (base 10). Value = mantissa * 10**exponent.
    """
    raw = int(raw_uint16) & 0xFFFF
    mantissa = raw & 0x0FFF
    exponent = (raw >> 12) & 0x0F
    # sign-extend 12-bit mantissa
    if mantissa & 0x800:
        mantissa -= 0x1000
    # sign-extend 4-bit exponent
    if exponent & 0x8:
        exponent -= 0x10
    return float(mantissa) * (10 ** exponent)


def parse_measurement(data: bytearray) -> dict:
    """Parse a BLE Blood Pressure Measurement characteristic payload.

    This function expects the payload layout used by many devices: flags byte
    followed by SFLOAT values for systolic/diastolic/mean arterial pressure.
    Pulse rate may be present at offset 14 as UINT16 depending on flags.
    """
    if not data or len(data) < 7:
        raise ValueError('data too short')
    flags = data[0]
    unit_is_mmhg = not (flags & 0x01)

    def _get_sfloat(off: int) -> float:
        raw = int.from_bytes(data[off:off+2], 'little')
        return _sfloat_to_float(raw)

    systolic = _get_sfloat(1)
    diastolic = _get_sfloat(3)
    mean_ap = _get_sfloat(5)

    pulse = None
    if len(data) >= 16:
        try:
            pulse = int.from_bytes(data[14:16], 'little')
        except Exception:
            pulse = None

    return {
        'systolic': systolic,
        'diastolic': diastolic,
        'mean_ap': mean_ap,
        'pulse': pulse,
        'unit': 'mmHg' if unit_is_mmhg else 'kPa'
    }


if __name__ == '__main__':
    # simple smoke test with a synthetic example
    # Example: mantissa=120, exponent=0 -> 120.0 ; encode as 12-bit mantissa + 4-bit exponent
    mantissa = 120 & 0x0FFF
    exponent = 0 & 0x0F
    raw = (exponent << 12) | mantissa
    b = raw.to_bytes(2, 'little')
    print('Raw bytes:', b.hex())
    print('Decoded sfloat:', _sfloat_to_float(int.from_bytes(b, 'little')))
