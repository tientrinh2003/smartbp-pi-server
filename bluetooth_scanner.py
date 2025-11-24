#!/usr/bin/env python3
"""
Bluetooth Scanner - Quét thiết bị BLE xung quanh
"""
import asyncio
from bleak import BleakScanner
import logging

logger = logging.getLogger(__name__)

async def scan_bluetooth_devices(scan_duration: int = 5):
    """
    Quét thiết bị Bluetooth xung quanh
    
    Args:
        scan_duration: Thời gian quét (giây)
    
    Returns:
        List[dict]: Danh sách thiết bị tìm thấy
    """
    try:
        logger.info(f"🔍 Bắt đầu quét Bluetooth trong {scan_duration}s...")
        devices = await BleakScanner.discover(timeout=scan_duration)
        
        result = []
        for device in devices:
            device_info = {
                "address": device.address,
                "name": device.name or "Unknown Device",
                "rssi": device.rssi,
                "is_omron": "OMRON" in (device.name or "").upper() or "BLE" in (device.name or "").upper()
            }
            result.append(device_info)
            logger.info(f"  📱 {device_info['name']} ({device_info['address']}) RSSI: {device_info['rssi']}")
        
        logger.info(f"✅ Tìm thấy {len(result)} thiết bị")
        return result
    
    except Exception as e:
        logger.error(f"❌ Lỗi quét Bluetooth: {e}")
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    devices = asyncio.run(scan_bluetooth_devices(10))
    print(f"\nTìm thấy {len(devices)} thiết bị:")
    for d in devices:
        print(f"  - {d['name']} | {d['address']} | RSSI: {d['rssi']}")
