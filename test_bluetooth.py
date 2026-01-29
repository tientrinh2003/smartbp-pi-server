#!/usr/bin/env python3
import asyncio
import logging
from bluetooth_bp_client import measure_once

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """
    Test kết nối Bluetooth và đo huyết áp
    """
    print("\n" + "="*60)
    print("TEST BLUETOOTH BLOOD PRESSURE")
    print("="*60 + "\n")
    
    # Địa chỉ máy Omron 
    DEVICE_ADDRESS = "CE:AC:75:17:41:17" 

    print(f"Thiết bị: {DEVICE_ADDRESS}")
    print(f"Timeout: 60 giây")
    print(f"Bước 1: Bật Bluetooth trên máy Omron")
    print(f"Bước 2: Đợi tới khi chương trình nói bấm START")
    print(f"Bước 3: Bấm nút START trên máy Omron\n")
    
    try:
        result = await measure_once(DEVICE_ADDRESS, timeout=60)
        
        if result:
            print("\n" + "="*60)
            print("THÀNH CÔNG!")
            print("="*60)
            print(f"Huyết áp: {result['sys']:.0f}/{result['dia']:.0f} {result['unit']}")
            print(f"Nhịp tim: {result.get('pulse', 'N/A')} bpm")
            if 'timestamp' in result:
                print(f"Thời gian: {result['timestamp']}")
            print("="*60 + "\n")
        else:
            print("\nKhông nhận được dữ liệu\n")
            
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
