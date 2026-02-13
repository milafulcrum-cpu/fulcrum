"""
Test version of ai_studio_code.py for Google Colab
- Uses dummy/mock data instead of camera/microphone
- Tests syntax and API integration without hardware
"""

!pip install google-genai pillow

import os
import asyncio
import base64
import io
import traceback
import numpy as np
from PIL import Image

from google import genai
from google.genai import types

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

# Check if API key exists
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("⚠️  WARNING: GEMINI_API_KEY not found!")
    print("Set it with: os.environ['GEMINI_API_KEY'] = 'your-key'")
else:
    print(f"✅ API Key found: {api_key[:10]}...")

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key or "dummy-key",
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)


class DummyFrameGenerator:
    """Generates fake frames instead of camera capture"""
    
    def __init__(self, mode="camera"):
        self.mode = mode
        self.frame_count = 0
    
    async def get_frame(self):
        """Generate a test frame (solid color with counter)"""
        await asyncio.sleep(1.0)  # Simulate 1 FPS
        
        # Create a simple test image
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[self.frame_count % 4]
        
        img = Image.new('RGB', (640, 480), color)
        
        # Add some text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        text = f"Frame {self.frame_count} - {self.mode}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        self.frame_count += 1
        
        # Convert to base64
        image_io = io.BytesIO()
        img.save(image_io, format='jpeg')
        image_io.seek(0)
        
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode()
        }


class DummyAudioGenerator:
    """Generates fake audio instead of microphone input"""
    
    def __init__(self):
        self.sample_count = 0
    
    async def read_audio(self):
        """Generate silent/dummy audio chunks"""
        await asyncio.sleep(0.064)  # ~16ms for 1024 samples at 16kHz
        
        # Silent PCM data (zeros)
        silent_data = b'\x00' * 2048  # 1024 samples * 2 bytes (16-bit)
        
        self.sample_count += 1024
        
        return {
            "data": silent_data,
            "mime_type": "audio/pcm"
        }


class AudioLoopTest:
    """Test version of AudioLoop without hardware dependencies"""
    
    def __init__(self, video_mode="camera"):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.frame_gen = DummyFrameGenerator(mode=video_mode)
        self.audio_gen = DummyAudioGenerator()
    
    async def send_text(self):
        """Simulated text input - auto-exits after 5 messages"""
        messages = [
            "Hello, testing!",
            "How are you?",
            "What's the weather?",
            "Tell me a joke",
            "q"  # Exit
        ]
        
        for msg in messages:
            await asyncio.sleep(3)
            print(f"[USER] {msg}")
            if msg.lower() == "q":
                break
            if self.session:
                await self.session.send(input=msg, end_of_turn=True)
    
    async def get_frames(self):
        """Generate test frames"""
        print(f"[CAMERA] Starting {self.video_mode} capture simulation...")
        
        while True:
            frame = await self.frame_gen.get_frame()
            if frame:
                print(f"[FRAME] Generated frame {self.frame_gen.frame_count}")
                await self.out_queue.put(frame)
            
            if self.frame_gen.frame_count >= 10:
                print("[CAMERA] Stopping after 10 frames")
                break
    
    async def send_realtime(self):
        """Send data to Gemini"""
        print("[SEND] Starting realtime stream...")
        
        while True:
            try:
                msg = await asyncio.wait_for(self.out_queue.get(), timeout=30)
                print(f"[SEND] Sending {msg['mime_type']} ({len(msg['data'])} bytes)")
                
                if self.session:
                    await self.session.send(input=msg)
            except asyncio.TimeoutError:
                print("[SEND] Queue timeout, stopping")
                break
    
    async def listen_audio(self):
        """Generate dummy audio"""
        print("[AUDIO IN] Starting dummy audio capture...")
        
        while True:
            audio_data = await self.audio_gen.read_audio()
            await self.out_queue.put(audio_data)
            
            if self.audio_gen.sample_count >= 16000 * 30:  # 30 seconds
                print("[AUDIO IN] Stopping after 30 seconds")
                break
    
    async def receive_audio(self):
        """Receive audio from Gemini"""
        print("[RECEIVE] Starting audio reception...")
        
        if not self.session:
            print("[RECEIVE] No session!")
            return
        
        try:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    audio_bytes = len(response.data)
                    print(f"[RECEIVE] Got audio: {audio_bytes} bytes")
                    self.audio_in_queue.put_nowait(response.data)
                
                if response.text:
                    print(f"[GEMINI] {response.text}")
        except Exception as e:
            print(f"[RECEIVE] Error: {e}")
    
    async def play_audio(self):
        """Simulate audio playback"""
        print("[PLAY] Starting audio playback simulation...")
        
        total_played = 0
        while True:
            try:
                bytestream = await asyncio.wait_for(
                    self.audio_in_queue.get(), 
                    timeout=5
                )
                total_played += len(bytestream)
                print(f"[PLAY] Played {len(bytestream)} bytes (total: {total_played})")
            except asyncio.TimeoutError:
                print("[PLAY] No audio received in 5 seconds")
                break
    
    async def run(self):
        """Main run loop"""
        print("=" * 50)
        print("TEST MODE - No real camera/microphone")
        print("=" * 50)
        
        if not api_key:
            print("\n❌ Cannot run without GEMINI_API_KEY")
            print("Add this cell before running:")
            print("import os")
            print("os.environ['GEMINI_API_KEY'] = 'YOUR_KEY_HERE'")
            return
        
        try:
            print("\n[INIT] Connecting to Gemini...")
            
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                print("[INIT] Connected! Starting tasks...\n")
                
                # Create tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                
                if self.video_mode in ["camera", "screen"]:
                    tg.create_task(self.get_frames())
                
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Wait for text task to complete
                await send_text_task
                
                print("\n[EXIT] User requested exit")
                raise asyncio.CancelledError("Test complete")
        
        except asyncio.CancelledError:
            print("\n✅ Test completed successfully!")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()


# Run the test
print("=" * 50)
print("GOOGLE COLAB TEST SCRIPT")
print("=" * 50)
print("\nThis script tests the structure without hardware.")
print("Set GEMINI_API_KEY and run: await main.run()")
print("\nTo run:")
print("  import os")
print("  os.environ['GEMINI_API_KEY'] = 'your-key'")
print("  main = AudioLoopTest()")
print("  await main.run()")

# Create instance
main = AudioLoopTest()
