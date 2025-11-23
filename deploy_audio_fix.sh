#!/bin/bash
# Deploy Audio Logic Fix to Raspberry Pi

echo "🔧 Deploying Audio Logic Fix..."

PI_HOST="192.168.22.70"
PI_USER="tien" 
PI_PATH="/home/tien/smartbp"

# Copy fixed file to Pi
echo "📤 Uploading fixed smartbp_pi5_enhanced.py..."
scp smartbp_pi5_enhanced.py ${PI_USER}@${PI_HOST}:${PI_PATH}/

echo "🔄 Restarting Pi server..."
ssh ${PI_USER}@${PI_HOST} << 'EOF'
cd /home/tien/smartbp
pkill -f smartbp_pi5_enhanced.py 2>/dev/null || true
sleep 2
source venv/bin/activate
nohup python3 smartbp_pi5_enhanced.py > server.log 2>&1 &
sleep 3
echo "✅ Pi server restarted with audio logic fix!"
EOF

echo "🧪 Testing fixed logic..."
sleep 5
curl -s http://${PI_HOST}:8000/api/ai/status | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f\"Speech: {data['speech']['status']} (conf: {data['speech']['confidence']})\")
    print(f\"Backend: {data['backend']}\")
    print('✅ Fixed API working!')
except:
    print('❌ API not responding yet, check server logs')
"

echo ""
echo "🎉 Audio Logic Fix Deployed!"
echo ""
echo "📋 What was fixed:"
echo "  ✓ Added proper handling for no audio input"
echo "  ✓ Set default no_speech state when no mic"
echo "  ✓ Prevent returning stale/fake audio data"
echo "  ✓ Added debug logging for audio buffer status"
echo ""
echo "💡 Now API will correctly return 'no_speech' when no audio input!"