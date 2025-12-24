#!/usr/bin/env python3
"""
Bluetooth Blood Pressure Client - Đo huyết áp một lần
"""
import asyncio
from bleak import BleakClient, BleakError
from datetime import datetime
import logging
from sfloat import parse_measurement  # Import IEEE 11073 SFLOAT decoder

logger = logging.getLogger(__name__)

BLOOD_PRESSURE_MEASUREMENT_UUID = "00002a35-0000-1000-8000-00805f9b34fb"

def parse_blood_pressure_data(data: bytearray) -> dict:
    """
    Phân tích dữ liệu BLE từ máy đo huyết áp theo chuẩn IEEE 11073 SFLOAT
    Returns: dict với sys, dia, map, pulse, timestamp
    """
    try:
        # Dùng SFLOAT decoder theo chuẩn IEEE 11073 (từ thesis Chapter 4)
        result = parse_measurement(data)
        
        # Parse timestamp nếu có (flags bit 1)
        flags = data[0]
        has_timestamp = flags & 0x02
        
        if has_timestamp and len(data) >= 14:
            idx = 7  # Timestamp starts after 3 SFLOAT values (sys, dia, map)
            year = int.from_bytes(data[idx:idx + 2], "little")
            month, day, hour, minute, second = data[idx + 2:idx + 7]
            result["timestamp"] = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        
        # Rename keys để tương thích với code cũ
        result["sys"] = int(result["systolic"])
        result["dia"] = int(result["diastolic"])
        result["map"] = int(result["mean_ap"])
        
        logger.info(f"✅ Đã đo (SFLOAT): SYS={result['sys']}, DIA={result['dia']}, Pulse={result.get('pulse', 'N/A')}")
        return result

    except Exception as e:
        logger.error(f"❌ Lỗi parse dữ liệu SFLOAT: {e}")
        return None


async def measure_once(device_address: str, timeout: int = 120) -> dict:
    """
    Kết nối và đo huyết áp một lần
    
    Args:
        device_address: Địa chỉ MAC của thiết bị
        timeout: Thời gian chờ tối đa (giây)
    
    Returns:
        dict: Kết quả đo MỚI NHẤT theo timestamp thực tế
    """
    all_measurements = []  # Lưu tất cả dữ liệu nhận được
    latest_result = None   # Kết quả mới nhất theo timestamp
    last_receive_time = [None]  # Thời gian nhận data cuối (dùng list để modify trong closure)
    connection_start_time = [None]  # Thời gian bắt đầu kết nối
    
    def notification_handler(sender, data):
        nonlocal all_measurements, latest_result
        parsed = parse_blood_pressure_data(data)
        if parsed:
            all_measurements.append(parsed)
            last_receive_time[0] = asyncio.get_event_loop().time()  # Cập nhật thời gian nhận
            
            # Tính thời gian từ lúc kết nối
            time_since_connect = asyncio.get_event_loop().time() - connection_start_time[0]
            
            logger.info(f"📥 Dữ liệu #{len(all_measurements)}: {parsed['sys']}/{parsed['dia']} mmHg (t+{time_since_connect:.1f}s)")
            
            # So sánh timestamp để tìm kết quả MỚI NHẤT (gần với thời gian hiện tại nhất)
            if 'timestamp' in parsed:
                if latest_result is None or parsed['timestamp'] > latest_result.get('timestamp', ''):
                    latest_result = parsed
                    logger.info(f"   ✅ Cập nhật kết quả mới nhất: {parsed['timestamp']}")
                else:
                    logger.info(f"   ⏭️ Kết quả cũ hơn: {parsed['timestamp']}")
            else:
                # Không có timestamp thì lấy kết quả cuối cùng
                if latest_result is None:
                    latest_result = parsed
                    logger.info(f"   ✅ Chấp nhận (no timestamp)")
    
    client = None
    try:
        logger.info(f"🔗 Đang kết nối tới {device_address}...")
        client = BleakClient(device_address, timeout=30.0)
        await client.connect()
        
        if not client.is_connected:
            logger.error("❌ Không thể kết nối")
            return None
        
        connection_start_time[0] = asyncio.get_event_loop().time()  # Lưu thời gian kết nối
        logger.info("✅ Đã kết nối, đang chờ dữ liệu (timeout: {}s)...".format(timeout))
        await client.start_notify(BLOOD_PRESSURE_MEASUREMENT_UUID, notification_handler)
        
        # Chờ nhận dữ liệu - giữ kết nối liên tục như code gốc
        start_time = asyncio.get_event_loop().time()
        no_data_timeout = 5  # Dừng nếu không có data mới trong 5 giây (sau khi đã có ít nhất 1 kết quả)
        
        while True:
            # Kiểm tra kết nối còn tồn tại không
            if not client.is_connected:
                logger.warning("⚠️ Thiết bị tự ngắt kết nối")
                break
            
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - start_time
            
            # Timeout tổng
            if elapsed > timeout:
                logger.warning(f"⏱️ Timeout tổng ({timeout}s), dừng nhận dữ liệu")
                break
            
            # Nếu đã nhận ít nhất 1 kết quả và không có data mới trong 5 giây → dừng
            if len(all_measurements) > 0 and last_receive_time[0] is not None:
                time_since_last = current_time - last_receive_time[0]
                if time_since_last > no_data_timeout:
                    logger.info(f"✅ Không còn data mới sau {no_data_timeout}s")
                    logger.info(f"   Tổng {len(all_measurements)} kết quả, thời gian: {int(elapsed)}s")
                    break
            
            await asyncio.sleep(1.0)  # Sleep 1s thay vì 0.5s để giảm CPU
        
        # Stop notify (nếu còn kết nối)
        if client.is_connected:
            await client.stop_notify(BLOOD_PRESSURE_MEASUREMENT_UUID)
            logger.info("🔌 Đã dừng nhận notification")
        
        # Trả về dữ liệu MỚI NHẤT theo timestamp
        if latest_result:
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f"✅ KẾT QUẢ ĐO HUYẾT ÁP MỚI NHẤT:")
            logger.info(f"   Huyết áp: {latest_result['sys']}/{latest_result['dia']} mmHg")
            logger.info(f"   Nhịp tim: {latest_result.get('pulse', 'N/A')} bpm")
            if 'timestamp' in latest_result:
                logger.info(f"   Thời gian: {latest_result['timestamp']}")
            logger.info(f"   (Tổng cộng nhận {len(all_measurements)} kết quả từ máy)")
            logger.info(f"{'='*60}")
            logger.info(f"")
            return latest_result
        else:
            logger.warning("⚠️ Không nhận được dữ liệu nào")
            return None
    
    except asyncio.TimeoutError:
        logger.error("⏱️ Timeout kết nối")
        return None
    except BleakError as e:
        logger.error(f"❌ Bluetooth error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return None
    finally:
        if client and client.is_connected:
            await client.disconnect()
            logger.info("🔌 Đã ngắt kết nối")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEVICE_ADDRESS = "00:5F:BF:3A:51:BD"
    result = asyncio.run(measure_once(DEVICE_ADDRESS, timeout=30))
    if result:
        print(f"\n📊 Kết quả đo:")
        print(f"  SYS: {result['sys']} {result['unit']}")
        print(f"  DIA: {result['dia']} {result['unit']}")
        print(f"  Pulse: {result.get('pulse', 'N/A')} bpm")
    else:
        print("❌ Đo thất bại")
