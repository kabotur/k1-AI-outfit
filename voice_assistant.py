"""
基于 K1 端侧 AI 的视觉生活小助手 - 单文件演示版
================================================

功能:
1. HDMI/桌面窗口实时预览 USB 摄像头画面。
2. 键盘触发:
   - O: 评价穿搭
   - Q/ESC: 退出
3. 可选语音触发:
   - 依赖 pyaudio + sensevoice。
   - 识别到“穿搭/搭配/衣服”触发穿搭评价。
4. 无模型时使用可解释的演示兜底规则，保证比赛现场先跑通闭环。
5. 有模型时可通过环境变量接入外部推理命令或 OpenCV DNN ONNX 模型。

K1 Bianbu OS 常用依赖:
    sudo apt update
    sudo apt install -y python3-opencv python3-numpy python3-pyaudio \
        espeak-ng alsa-utils pulseaudio-utils python3-spacemit-ort

运行:
    python3 voice_assistant.py

可选模型接入:
    export OUTFIT_MODEL=/path/to/outfit.onnx

也可以接外部 NPU 推理程序，命令最后会自动追加图片路径:
    export OUTFIT_INFER_CMD="/path/to/outfit_infer"
外部程序输出 JSON 最佳:
    {"label":"搭配优秀","confidence":0.92}
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - target board dependency
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - target board dependency
    np = None

try:
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover - optional voice trigger
    pyaudio = None

try:
    import audioop  # type: ignore
except Exception:  # pragma: no cover - Python 3.13 removed audioop
    audioop = None

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover - optional Chinese overlay
    Image = None
    ImageDraw = None
    ImageFont = None


# ======================= 配置 =======================
ROOT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT_DIR / "captures"


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value not in ("0", "false", "False", "no", "No")


def _first_existing_dir(paths: list[str]) -> str:
    for path in paths:
        expanded = os.path.expanduser(path)
        if expanded and Path(expanded).is_dir():
            return expanded
    return os.path.expanduser(paths[0]) if paths else ""


def _clean_play_device(value: str) -> str:
    value = (value or "").strip()
    for marker in (
        "TTS_",
        "PLAY_",
        "MIC_",
        "ENABLE_",
        "SAY_",
        "SPACEMIT_",
        "CAMERA_",
    ):
        idx = value.find(marker)
        if idx > 0:
            cleaned = value[:idx].strip()
            print(f"[audio] 修正 PLAY_DEVICE: {value} -> {cleaned}")
            return cleaned
    return value


def _normalize_voice_backend(value: str) -> str:
    backend = (value or "").strip().lower()
    if not backend and os.environ.get("ARECORD_DEVICE", "").strip():
        print("[voice] 检测到 ARECORD_DEVICE，自动使用 VOICE_BACKEND=arecord")
        return "arecord"
    if not backend:
        backend = "pyaudio"
    aliases = {
        "alsa": "arecord",
        "record": "arecord",
        "rec": "arecord",
    }
    normalized = aliases.get(backend, backend)
    if normalized != backend:
        print(f"[voice] 修正 VOICE_BACKEND: {backend} -> {normalized}")
    return normalized


CAMERA_INDEX_ENV = os.environ.get("CAMERA_INDEX")
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0) or 0
CAMERA_AUTO_FALLBACK = _env_bool("CAMERA_AUTO_FALLBACK", True)
CAMERA_SCAN_MAX = max(0, _env_int("CAMERA_SCAN_MAX", 6) or 0)
CAMERA_READ_RETRIES = max(1, _env_int("CAMERA_READ_RETRIES", 5) or 5)
CAMERA_RECONNECT_FAILURES = max(1, _env_int("CAMERA_RECONNECT_FAILURES", 20) or 20)
CAMERA_RECONNECT_SECONDS = max(
    0.1, float(os.environ.get("CAMERA_RECONNECT_SECONDS", "2.0"))
)
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
CAMERA_FOURCC = os.environ.get("CAMERA_FOURCC", "MJPG").strip()
CAMERA_BUFFER_SIZE = max(1, _env_int("CAMERA_BUFFER_SIZE", 1) or 1)
CAMERA_STABLE_INDEX_ENV = os.environ.get("CAMERA_STABLE_INDEX")
CAMERA_STABLE_INDEX = _env_int("CAMERA_STABLE_INDEX")
RESULT_HOLD_SECONDS = float(os.environ.get("RESULT_HOLD_SECONDS", "5"))

SENSEVOICE_MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR") or _first_existing_dir(
    [
        "/usr/share/spacemit-asr/sensevoice",
        "~/.cache/models/asr/sensevoice",
        "~/.cache/sensevoice",
    ]
)
SPACEMIT_ASR_PATHS = [
    path
    for path in os.environ.get(
        "SPACEMIT_ASR_PATHS",
        "/home/bianbu/spacemit-demo/examples/NLP:/opt/spacemit-asr:~/spacemit-demo/examples/NLP",
    ).split(":")
    if path
]
ENABLE_VOICE = os.environ.get("ENABLE_VOICE", "1") not in ("0", "false", "False")
CAMERA_STARTUP = _env_bool("CAMERA_STARTUP", False)
CAMERA_CLOSE_AFTER_ACTION = _env_bool("CAMERA_CLOSE_AFTER_ACTION", True)
ACTION_PREVIEW_SECONDS = max(0.0, float(os.environ.get("ACTION_PREVIEW_SECONDS", "2.0")))
ENABLE_ASR = os.environ.get("ENABLE_ASR", "1") not in ("0", "false", "False")
VOICE_RMS_THRESHOLD = int(os.environ.get("VOICE_RMS_THRESHOLD", "120"))
VOICE_COMMAND_SECONDS = float(os.environ.get("VOICE_COMMAND_SECONDS", "3.0"))
VOICE_COOLDOWN_SECONDS = float(os.environ.get("VOICE_COOLDOWN_SECONDS", "4.0"))
VOICE_FALLBACK_ACTION = os.environ.get("VOICE_FALLBACK_ACTION", "").strip().lower()
ENABLE_VOICE_QA = _env_bool("ENABLE_VOICE_QA", False)
VOICE_DEBUG = _env_bool("VOICE_DEBUG", True)
VOICE_PAUSE_AFTER_TTS = float(os.environ.get("VOICE_PAUSE_AFTER_TTS", "6.0"))
VOICE_SELF_ECHO_SECONDS = float(os.environ.get("VOICE_SELF_ECHO_SECONDS", "25.0"))
VOICE_SAMPLE_RATE_ENV = os.environ.get("VOICE_SAMPLE_RATE")
VOICE_SAMPLE_RATE = max(8000, _env_int("VOICE_SAMPLE_RATE", 44100) or 44100)
VOICE_ZERO_INPUT_WARN_SECONDS = max(5.0, float(os.environ.get("VOICE_ZERO_INPUT_WARN_SECONDS", "10.0")))
MIC_INDEX_ENV = os.environ.get("MIC_INDEX")
MIC_INDEX = _env_int("MIC_INDEX")
VOICE_BACKEND = _normalize_voice_backend(os.environ.get("VOICE_BACKEND", ""))
ARECORD_DEVICE = os.environ.get("ARECORD_DEVICE", "").strip()
ARECORD_PROBE_SECONDS = max(0.2, float(os.environ.get("ARECORD_PROBE_SECONDS", "0.6")))
VOICE_AUTO_ARECORD_FALLBACK = _env_bool("VOICE_AUTO_ARECORD_FALLBACK", True)

PLAY_CMD_OVERRIDE = os.environ.get("PLAY_CMD", "aplay")
PLAY_DEVICE = _clean_play_device(os.environ.get("PLAY_DEVICE", "plughw:2,0"))
PLAY_ALLOW_CARD0_FALLBACK = _env_bool("PLAY_ALLOW_CARD0_FALLBACK", False)
PLAY_CANDIDATES = ["aplay", "paplay", "pw-play"]
ENABLE_TTS = _env_bool("ENABLE_TTS", True)
SAY_STARTUP = _env_bool("SAY_STARTUP", False)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto").strip().lower()
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "zh").strip() or "zh"
TTS_ALLOW_ESPEAK_ZH = _env_bool("TTS_ALLOW_ESPEAK_ZH", False)
TTS_STREAM_CHUNKS = _env_bool("TTS_STREAM_CHUNKS", True)
TTS_CHUNK_CHARS = max(1, int(os.environ.get("TTS_CHUNK_CHARS", "18")))
SPACEMIT_DEMO_NLP_DIR = Path(
    os.path.expanduser(os.environ.get("SPACEMIT_DEMO_NLP_DIR", "~/spacemit-demo/examples/NLP"))
)
SHOW_WINDOW = _env_bool("SHOW_WINDOW", True)
WINDOW_NAME = os.environ.get("WINDOW_NAME", "K1 Life Assistant")

OUTFIT_LABELS = ["搭配优秀", "搭配一般", "搭配需要调整"]

COLOR_SCORE_STANDARD_REPLY = (
    "穿搭颜色评分标准是：同色系协调加分，中性色过渡加分；"
    "上浅下深或上深下浅有层次加分；"
    "亮冷色和亮暖色大面积混搭会扣分。"
    "评分参考是优秀88分，一般72分，需要调整58分。"
)
SPECIAL_VOICE_REPLIES: list[tuple[str, tuple[str, ...], str]] = [
    ("蓝色衣服", ("蓝色", "蓝衣", "蓝色衣服", "蓝色的衣服"), "蓝色衣服在1专区。"),
    ("红色衣服", ("红色", "红衣", "红色衣服", "红色的衣服"), "红色衣服在2专区。"),
    ("黑色衣服", ("黑色", "黑衣", "黑色衣服", "黑色的衣服"), "黑色衣服在3专区。"),
    ("白色衣服", ("白色", "白衣", "白色衣服", "白色的衣服"), "白色衣服在4专区。"),
]
SPECIAL_LOCATION_WORDS = ("在哪", "哪里", "哪儿", "几区", "几专区", "专区", "位置", "怎么找")
COLOR_SCORE_STANDARD_WORDS = (
    "评分标准",
    "颜色标准",
    "颜色评分",
    "配色标准",
    "怎么评分",
    "怎么打分",
    "按什么评分",
    "按什么打分",
)

_tts_model = None
_tts_playing = threading.Event()
_voice_pause_until = 0.0
_last_tts_text = ""
_last_tts_time = 0.0


@dataclass
class AppResult:
    title: str
    label: str
    confidence: float
    speech: str
    details: str
    mode: str
    image_path: Optional[Path] = None


@dataclass
class ColorStats:
    name: str
    tone: str
    brightness: str
    hue_deg: float
    saturation: float
    value: float

    @property
    def is_large_bright(self) -> bool:
        return self.saturation >= 55 and self.value >= 135


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _display_available() -> bool:
    if os.name == "nt":
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    probes = [
        ["xdpyinfo", "-display", display],
        ["xset", "-display", display, "q"],
    ]
    for probe in probes:
        if not _have(probe[0]):
            continue
        try:
            proc = subprocess.run(
                probe,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return True
    print(f"[display] DISPLAY={display} 无法连接或无 X11 授权，自动关闭窗口")
    return False


def _camera_device_label(index: int) -> str:
    if os.name == "nt":
        return f"camera index {index}"
    return f"/dev/video{index}"


def _video_device_name(index: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        return name_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _camera_priority(index: int) -> tuple[int, int]:
    name = _video_device_name(index).lower()
    if "metadata" in name or "meta" in name:
        return (8, index)
    if "usb" in name or "uvc" in name:
        return (0, index)
    if "camera" in name:
        return (1, index)
    if "vivi" in name:
        return (7, index)
    if Path(f"/dev/video{index}").exists():
        return (2, index)
    return (5, index)


def _dedupe_indexes(indexes: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for index in indexes:
        if index < 0 or index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


def _candidate_camera_indexes() -> list[int]:
    discovered = _discover_camera_indexes()
    sorted_discovered = sorted(discovered, key=_camera_priority)
    stable = [CAMERA_STABLE_INDEX] if CAMERA_STABLE_INDEX is not None else []
    if CAMERA_INDEX_ENV not in (None, ""):
        if not CAMERA_AUTO_FALLBACK:
            return [CAMERA_INDEX]
        # USB cameras often expose two neighboring nodes: one image stream and one metadata node.
        preferred = [CAMERA_INDEX, CAMERA_INDEX + 1, CAMERA_INDEX - 1]
        return _dedupe_indexes(stable + preferred + sorted_discovered)
    return _dedupe_indexes(stable + sorted_discovered)


def _create_capture(index: int):
    if os.name != "nt" and hasattr(cv2, "CAP_V4L2"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def _read_valid_frame(cap) -> tuple[bool, object | None]:
    for _ in range(CAMERA_READ_RETRIES):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(0.08)
    return False, None


def _open_camera() -> tuple[object, int] | tuple[None, None]:
    indexes = _candidate_camera_indexes()
    if CAMERA_INDEX_ENV not in (None, "") and CAMERA_AUTO_FALLBACK:
        print(
            f"[camera] 优先尝试 CAMERA_INDEX={CAMERA_INDEX}，失败后自动扫描其他节点"
        )
    else:
        print(f"[camera] 自动检测候选节点: {', '.join(map(str, indexes[:20]))}")

    for index in indexes:
        label = _camera_device_label(index)
        name = _video_device_name(index)
        name_hint = f" ({name})" if name else ""
        cap = _create_capture(index)
        if not cap.isOpened():
            cap.release()
            print(f"[camera] {label}{name_hint} 不可用，继续尝试")
            continue
        if CAMERA_FOURCC:
            fourcc = cv2.VideoWriter_fourcc(*CAMERA_FOURCC[:4].ljust(4))
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        ok, frame = _read_valid_frame(cap)
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"[camera] {label}{name_hint} 读帧成功: {w}x{h}")
            return cap, index
        cap.release()
        print(f"[camera] {label}{name_hint} 不是可读图像流或无法读帧，继续尝试")
    return None, None


def _discover_camera_indexes() -> list[int]:
    indexes = set(range(CAMERA_SCAN_MAX + 1))
    for device in Path("/dev").glob("video*"):
        suffix = device.name.replace("video", "", 1)
        if suffix.isdigit():
            indexes.add(int(suffix))
    return sorted(indexes)


def _compute_rms(pcm_bytes: bytes, sampwidth: int = 2) -> int:
    if not pcm_bytes:
        return 0
    if audioop is not None:
        try:
            return int(audioop.rms(pcm_bytes, sampwidth))
        except Exception:
            pass
    if len(pcm_bytes) < 2:
        return 0
    import struct

    n = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    return int(math.sqrt(sum(s * s for s in samples) / max(n, 1)))


def _input_devices(p) -> list[dict]:
    devices: list[dict] = []
    for index in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(index)
        except Exception:
            continue
        channels = int(info.get("maxInputChannels", 0) or 0)
        if channels <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": str(info.get("name", "?")),
                "channels": channels,
                "rate": info.get("defaultSampleRate", "?"),
            }
        )
    return devices


def _input_device_priority(device: dict) -> tuple[int, int]:
    name = str(device.get("name", "")).lower()
    index = int(device.get("index", 9999))
    if "usb pnp" in name or "pnp sound" in name:
        return (0, index)
    if "usb audio" in name:
        return (1, index)
    if "usb" in name:
        return (2, index)
    if "default" in name:
        return (3, index)
    if "pipewire" in name or "pulse" in name:
        return (4, index)
    return (5, index)


def _select_fallback_input_device(devices: list[dict]) -> Optional[int]:
    if not devices:
        return None
    return int(sorted(devices, key=_input_device_priority)[0]["index"])


def _is_soft_audio_input(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("pipewire", "pulse", "default", "jack"))


def _print_zero_input_hint(device_name: str, rate: int) -> None:
    print(
        "[voice] 连续检测到 RMS=0，当前 PyAudio 输入可能没有接到真实麦克风: "
        f"{device_name}"
    )
    if _is_soft_audio_input(device_name):
        print("[voice] 现场建议先执行: arecord -l")
        print(
            "[voice] 如果能看到 USB PnP Sound Device，改用: "
            f"VOICE_BACKEND=arecord ARECORD_DEVICE=plughw:卡号,0 VOICE_SAMPLE_RATE={rate} "
            "~/spacemit-demo/examples/NLP/.venv/bin/python ~/桌面/voice_assistant.py"
        )
    else:
        print("[voice] 请确认麦克风未静音、插入的是录音口，并尝试降低 VOICE_RMS_THRESHOLD。")


def _voice_sample_rate_candidates(default_rate: Optional[int]) -> list[int]:
    candidates: list[int] = []
    if VOICE_SAMPLE_RATE:
        candidates.append(VOICE_SAMPLE_RATE)
    if default_rate:
        candidates.append(default_rate)
    candidates.extend([44100, 48000, 16000, 32000, 8000])
    unique: list[int] = []
    for rate in candidates:
        try:
            clean_rate = int(rate)
        except (TypeError, ValueError):
            continue
        if clean_rate >= 8000 and clean_rate not in unique:
            unique.append(clean_rate)
    return unique


def _print_voice_backend_fallback_hint(rate: int) -> None:
    print("[voice] PyAudio 输入流无法启动，可改用 ALSA 直连录音后端排查")
    print("[voice] 先查看真实录音设备: arecord -l")
    print(
        "[voice] 如果能看到 USB 麦克风，先测试: "
        f"arecord -D plughw:卡号,0 -f S16_LE -c1 -r {rate} -t wav -d 5 /tmp/mic.wav"
    )
    print(
        "[voice] 测试有声音后启动: "
        f"VOICE_BACKEND=arecord ARECORD_DEVICE=plughw:卡号,0 VOICE_SAMPLE_RATE={rate} "
        "~/spacemit-demo/examples/NLP/.venv/bin/python ~/桌面/voice_assistant.py"
    )


def _default_arecord_device_from_aplay() -> str:
    match = re.search(r"plughw:(\d+),", PLAY_DEVICE or "")
    if not match:
        return ""
    card = int(match.group(1))
    if card > 0:
        return f"plughw:{card + 1},0"
    return ""


def _print_tts_dependency_hint(exc: Exception) -> None:
    message = str(exc)
    if "soundfile" in message:
        print("[tts] 中文 TTS 缺少 soundfile，当前会影响 spacemit_tts 合成")
        print(
            "[tts] 修复命令: "
            "~/spacemit-demo/examples/NLP/.venv/bin/python -m pip install soundfile"
        )
    elif "cn2an" in message:
        print("[tts] 中文 TTS 缺少 cn2an")
        print(
            "[tts] 修复命令: "
            "~/spacemit-demo/examples/NLP/.venv/bin/python -m pip install cn2an"
        )


def _resolve_input_device_index(p) -> Optional[int]:
    devices = _input_devices(p)
    available = {int(device["index"]) for device in devices}
    if MIC_INDEX is not None:
        if MIC_INDEX in available:
            return MIC_INDEX
        fallback = _select_fallback_input_device(devices)
        if fallback is None:
            print(f"[warn] MIC_INDEX={MIC_INDEX} 不存在或不是输入设备，未发现可用麦克风")
            return None
        print(f"[warn] MIC_INDEX={MIC_INDEX} 不存在或不是输入设备，自动改用 MIC_INDEX={fallback}")
        return fallback
    return _select_fallback_input_device(devices)


def _device_default_sample_rate(p, index: Optional[int]) -> Optional[int]:
    if index is None:
        return None
    try:
        rate = p.get_device_info_by_index(index).get("defaultSampleRate")
        return int(float(rate))
    except Exception:
        return None


def _print_input_devices() -> None:
    if pyaudio is None:
        return
    p = pyaudio.PyAudio()
    try:
        print("[voice] 可用输入设备:")
        devices = _input_devices(p)
        for device in devices:
            print(
                f"  MIC_INDEX={device['index']}: {device['name']} | "
                f"输入通道={device['channels']} | 默认采样率={device['rate']}"
            )
        if not devices:
            print("  未发现可用输入设备，请检查麦克风连接和系统录音权限")
            return
        if MIC_INDEX is None:
            fallback = _select_fallback_input_device(devices)
            if fallback is None:
                print("  当前未指定 MIC_INDEX，将使用系统默认输入设备")
            else:
                print(f"  当前未指定 MIC_INDEX，将优先使用系统默认输入；必要时可试 MIC_INDEX={fallback}")
        elif MIC_INDEX not in {int(device["index"]) for device in devices}:
            fallback = _select_fallback_input_device(devices)
            if fallback is not None:
                print(f"  当前 MIC_INDEX={MIC_INDEX} 不存在，运行时将自动改用 MIC_INDEX={fallback}")
    finally:
        p.terminate()


def _pick_player() -> Optional[str]:
    if PLAY_CMD_OVERRIDE:
        if _have(PLAY_CMD_OVERRIDE):
            return PLAY_CMD_OVERRIDE
        print(f"[audio] PLAY_CMD={PLAY_CMD_OVERRIDE} 找不到，改用自动选择")
    for cmd in PLAY_CANDIDATES:
        if _have(cmd):
            return cmd
    return None


def _discover_aplay_devices() -> list[tuple[str, int, str]]:
    if not _have("aplay"):
        return []
    try:
        proc = subprocess.run(
            ["aplay", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    devices: list[tuple[str, int, str]] = []
    for line in proc.stdout.splitlines():
        match = re.search(r"card\s+(\d+):\s*([^,]+),.*device\s+(\d+):", line)
        if not match:
            continue
        card = int(match.group(1))
        name = match.group(2).strip()
        device = f"plughw:{card},{match.group(3)}"
        if card == 0 and not PLAY_ALLOW_CARD0_FALLBACK:
            continue
        if not any(item[0] == device for item in devices):
            devices.append((device, card, name))
    devices.sort(key=_aplay_device_priority)
    return devices


def _aplay_device_priority(item: tuple[str, int, str]) -> tuple[int, int, str]:
    _device, card, name = item
    lowered = name.lower()
    if "usb" in lowered or "pnp" in lowered:
        return (0, card, name)
    if card == 0:
        return (9, card, name)
    return (3, card, name)


def _aplay_device_candidates() -> list[Optional[str]]:
    candidates: list[Optional[str]] = []
    if PLAY_DEVICE:
        candidates.append(PLAY_DEVICE)
    for device, _card, _name in _discover_aplay_devices():
        if device not in candidates:
            candidates.append(device)
    return candidates


def _play_wav(wav_path: Path) -> bool:
    player = _pick_player()
    if player is None:
        print("[tts] 未找到 paplay/pw-play/aplay，跳过播放")
        return False

    if player == "aplay":
        for device in _aplay_device_candidates():
            cmd = [player]
            device_hint = ""
            if device:
                cmd.extend(["-D", device])
                device_hint = f" PLAY_DEVICE={device}"
            cmd.append(str(wav_path))
            try:
                proc = subprocess.run(cmd, check=False)
            except OSError as exc:
                print(f"[tts] 执行播放器失败: {exc}")
                return False
            if proc.returncode == 0:
                if device and device != PLAY_DEVICE:
                    print(f"[tts] 自动切换播放设备: {device}")
                return True
            print(f"[tts] {player}{device_hint} 播放失败，返回码 {proc.returncode}")
        return False

    try:
        proc = subprocess.run([player, str(wav_path)], check=False)
    except OSError as exc:
        print(f"[tts] 执行播放器失败: {exc}")
        return False
    if proc.returncode != 0:
        print(f"[tts] {player} 播放失败，返回码 {proc.returncode}")
        return False
    return True


def _prepare_spacemit_tts_import() -> None:
    nlp_dir = str(SPACEMIT_DEMO_NLP_DIR)
    if SPACEMIT_DEMO_NLP_DIR.is_dir() and nlp_dir not in sys.path:
        sys.path.insert(0, nlp_dir)


def _spacemit_tts_available() -> bool:
    if TTS_ENGINE in ("espeak", "espeak-ng"):
        return False
    try:
        _prepare_spacemit_tts_import()
        from spacemit_tts.melotts.melotts_onnx import TTSModel  # noqa: F401

        return True
    except Exception:
        return False


def _get_spacemit_tts_model():
    global _tts_model
    if _tts_model is None:
        _prepare_spacemit_tts_import()
        from spacemit_tts.melotts.melotts_onnx import TTSModel  # type: ignore

        print("[tts] 加载 spacemit_tts 模型...")
        _tts_model = TTSModel()
        print("[tts] spacemit_tts 模型就绪")
    return _tts_model


def _split_tts_chunks(text: str) -> list[str]:
    if not TTS_STREAM_CHUNKS:
        return [text]
    chunks: list[str] = []
    buf = ""
    for char in text:
        buf += char
        if char in "，。！？；,.!?;" or len(buf) >= TTS_CHUNK_CHARS:
            chunk = buf.strip()
            if chunk:
                chunks.append(chunk)
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [text]


def _normalize_voice_text(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _is_probable_self_echo(text: str) -> bool:
    if not text or not _last_tts_text:
        return False
    if time.time() - _last_tts_time > VOICE_SELF_ECHO_SECONDS:
        return False
    recognized = _normalize_voice_text(text)
    spoken = _normalize_voice_text(_last_tts_text)
    if len(recognized) < 4 or len(spoken) < 4:
        return False
    if recognized in spoken or spoken in recognized:
        return True
    common = set(recognized) & set(spoken)
    overlap = len(common) / max(1, min(len(set(recognized)), len(set(spoken))))
    return overlap >= 0.72


def _tts_play_spacemit(text: str) -> bool:
    if TTS_ENGINE in ("espeak", "espeak-ng"):
        return False
    try:
        model = _get_spacemit_tts_model()
        for chunk in _split_tts_chunks(text):
            wav_path = Path(str(model.ort_predict(chunk)))
            print(f"[tts] 生成音频: {wav_path}")
            _play_wav(wav_path)
        return True
    except Exception as exc:
        print(f"[tts] spacemit_tts 不可用: {exc}")
        _print_tts_dependency_hint(exc)
        return TTS_ENGINE in ("spacemit", "spacemit_tts")


def _tts_play_espeak(text: str) -> None:
    if TTS_LANGUAGE.lower().startswith("zh") and not TTS_ALLOW_ESPEAK_ZH:
        print("[tts] 中文播报未使用 espeak 兜底，避免中文发音异常")
        print("[tts] 如只想测试音频链路，可临时设置 TTS_ALLOW_ESPEAK_ZH=1")
        return
    espeak_cmd = "espeak-ng" if _have("espeak-ng") else ("espeak" if _have("espeak") else None)
    if espeak_cmd is None:
        print("[tts] 未找到 espeak-ng/espeak，跳过语音播报")
        return

    wav_path = Path(tempfile.gettempdir()) / "k1_life_assistant_reply.wav"
    try:
        if wav_path.exists():
            wav_path.unlink()
        proc = subprocess.run(
            [espeak_cmd, "-v", TTS_LANGUAGE, "-s", "150", "-w", str(wav_path), text],
            check=False,
        )
        if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
            print(f"[tts] {espeak_cmd} 生成 wav 失败，返回码 {proc.returncode}")
            return
        _play_wav(wav_path)
    except OSError as exc:
        print(f"[tts] 播报失败: {exc}")


def tts_play(text: str) -> None:
    """优先使用 K1 官方 spacemit_tts，缺失时退回 espeak/espeak-ng。"""
    global _voice_pause_until, _last_tts_text, _last_tts_time
    text = (text or "").strip()
    if not text:
        return
    print(f"[tts] {text}")
    if not ENABLE_TTS:
        print("[tts] ENABLE_TTS=0，跳过语音播报")
        return

    _tts_playing.set()
    try:
        _last_tts_text = text
        _last_tts_time = time.time()
        if _tts_play_spacemit(text):
            return
        _tts_play_espeak(text)
    finally:
        _voice_pause_until = time.time() + max(0.0, VOICE_PAUSE_AFTER_TTS)
        _tts_playing.clear()


class ExternalClassifier:
    """接入外部推理程序，适合后续替换为 K1 NPU 编译后的推理命令。"""

    def __init__(self, env_name: str, labels: list[str]) -> None:
        self.command = os.environ.get(env_name, "").strip()
        self.labels = labels

    @property
    def available(self) -> bool:
        return bool(self.command)

    def predict(self, image_path: Path) -> Optional[tuple[str, float, str]]:
        if not self.command:
            return None
        cmd = shlex.split(self.command) + [str(image_path)]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
        except Exception as exc:
            print(f"[model] 外部推理失败: {exc}")
            return None
        if proc.returncode != 0:
            print(f"[model] 外部推理返回码 {proc.returncode}: {proc.stderr.strip()}")
            return None
        raw = proc.stdout.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            label = str(data.get("label", "")).strip()
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            parts = raw.replace(",", " ").split()
            label = parts[0] if parts else ""
            confidence = float(parts[1]) if len(parts) > 1 and _is_float(parts[1]) else 0.85
        if label not in self.labels and self.labels:
            print(f"[model] 外部推理标签不在预设列表中: {label}")
        return label, max(0.0, min(confidence, 1.0)), "external"


class OpenCVDNNClassifier:
    """ONNX 分类模型兜底接入。K1 NPU 正式部署时建议用 ExternalClassifier 对接。"""

    def __init__(
        self,
        model_env: str,
        labels_env: str,
        default_labels: list[str],
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.model_path = os.environ.get(model_env, "").strip()
        self.labels = _load_labels(os.environ.get(labels_env, ""), default_labels)
        self.input_size = input_size
        self.net = None
        if self.model_path and cv2 is not None and Path(self.model_path).exists():
            try:
                self.net = cv2.dnn.readNet(self.model_path)
                print(f"[model] 已加载 ONNX 模型: {self.model_path}")
            except Exception as exc:
                print(f"[model] 加载 ONNX 模型失败: {exc}")

    @property
    def available(self) -> bool:
        return self.net is not None and np is not None and cv2 is not None

    def predict(self, frame) -> Optional[tuple[str, float, str]]:
        if not self.available:
            return None
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward().reshape(-1).astype("float32")
        probs = _softmax(out)
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, float(probs[idx]), "opencv-dnn"


class ClassifierHub:
    def __init__(self) -> None:
        self.outfit_external = ExternalClassifier("OUTFIT_INFER_CMD", OUTFIT_LABELS)
        self.outfit_dnn = OpenCVDNNClassifier("OUTFIT_MODEL", "OUTFIT_LABELS", OUTFIT_LABELS)

    def classify_outfit(self, frame, image_path: Path) -> AppResult:
        predicted = self.outfit_external.predict(image_path) or self.outfit_dnn.predict(frame)
        if predicted:
            label, confidence, mode = predicted
            speech = _outfit_speech(label, "模型推理结果")
            return AppResult(
                title="穿搭智能评价",
                label=label,
                confidence=confidence,
                speech=speech,
                details="模型推理结果",
                mode=mode,
                image_path=image_path,
            )
        return analyze_outfit_demo(frame, image_path)


class VoiceTrigger:
    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._asr_model = None
        self._asr_unavailable_reason: Optional[str] = None
        self._asr_unavailable_logged = False

    @property
    def enabled(self) -> bool:
        if not ENABLE_VOICE:
            return False
        if VOICE_BACKEND == "arecord":
            return _have("arecord")
        return pyaudio is not None

    def start(self) -> None:
        if not self.enabled:
            print("[voice] 未启用语音触发，使用键盘 O/Q 控制")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[voice] 语音触发线程已启动")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def get_event(self) -> Optional[tuple[str, str]]:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        if VOICE_BACKEND == "arecord":
            self._run_arecord()
            return
        if self._run_pyaudio():
            return
        if VOICE_AUTO_ARECORD_FALLBACK and _have("arecord"):
            print("[voice] PyAudio 不可用，自动切换到 arecord ALSA 直连录音")
            self._run_arecord()
            return
        print("[voice] 语音触发不可用，仍可使用键盘 O 触发穿搭评价")

    def _handle_voice_wav(self, wav_path: Path, fallback_text: str = "检测到人声") -> None:
        print(f"[voice] 录音文件: {wav_path}")
        text = self._recognize_voice(wav_path)
        print(f"[voice] ASR文本: {text!r}")
        if _is_probable_self_echo(text):
            print(f"[voice] 忽略疑似助手自播回声: {text!r}")
            return
        action = route_voice_text(text)
        if action is None and VOICE_FALLBACK_ACTION == "outfit":
            action = VOICE_FALLBACK_ACTION
        if action is None and ENABLE_VOICE_QA and text:
            action = "qa"
        if action:
            self.events.put((action, text or fallback_text))
        else:
            print(f"[voice] 检测到人声，但未识别到有效口令: {text!r}")

    def _run_arecord(self) -> None:
        rate = VOICE_SAMPLE_RATE or 44100
        probe_path = _voice_temp_wav("k1_life_assistant_voice_probe.wav")
        command_path = _voice_temp_wav("k1_life_assistant_voice_cmd.wav")
        last_trigger = 0.0
        last_rms_log = 0.0
        device_hint = ARECORD_DEVICE or _default_arecord_device_from_aplay() or "default"
        print(f"[voice] 使用 arecord ALSA 直连录音: {device_hint}")
        print(f"[voice] 录音采样率: {rate}Hz")
        try:
            while not self._stop.is_set():
                if _tts_playing.is_set() or time.time() < _voice_pause_until:
                    time.sleep(0.05)
                    continue
                if not _run_arecord(probe_path, ARECORD_PROBE_SECONDS, rate, device_hint):
                    time.sleep(1.0)
                    continue
                data, _actual_rate = _read_wav_pcm(probe_path)
                rms = _compute_rms(data)
                now = time.time()
                if VOICE_DEBUG and now - last_rms_log >= 2.0:
                    print(f"[voice] RMS={rms} 阈值={VOICE_RMS_THRESHOLD}")
                    last_rms_log = now
                if rms < VOICE_RMS_THRESHOLD or now - last_trigger < VOICE_COOLDOWN_SECONDS:
                    continue
                last_trigger = now
                print(f"[voice] 触发录音: RMS={rms}，录音 {VOICE_COMMAND_SECONDS:.1f}s")
                if not _run_arecord(command_path, VOICE_COMMAND_SECONDS, rate, device_hint):
                    continue
                self._handle_voice_wav(command_path)
        except Exception as exc:
            print(f"[voice] arecord 语音触发不可用: {exc}")

    def _run_pyaudio(self) -> bool:
        p = pyaudio.PyAudio()
        stream = None
        chunk = 1024
        last_trigger = 0.0
        last_rms_log = 0.0
        zero_input_since: Optional[float] = None
        zero_input_hint_printed = False
        try:
            input_kwargs = {}
            mic_index = _resolve_input_device_index(p)
            default_rate = _device_default_sample_rate(p, mic_index)
            dev_name = "系统默认输入设备"
            if mic_index is not None:
                input_kwargs["input_device_index"] = mic_index
                try:
                    dev_name = p.get_device_info_by_index(mic_index).get("name", "?")
                except Exception:
                    dev_name = "?"
                print(f"[voice] 使用麦克风 [{mic_index}] {dev_name}")
            else:
                print("[voice] 未指定有效 MIC_INDEX，使用系统默认输入设备")
            rate_candidates = _voice_sample_rate_candidates(default_rate)
            stream_error: Optional[Exception] = None
            rate = rate_candidates[0] if rate_candidates else 44100
            print(
                "[voice] 录音采样率候选: "
                + ", ".join(f"{candidate}Hz" for candidate in rate_candidates)
            )
            for candidate in rate_candidates:
                try:
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=candidate,
                        input=True,
                        frames_per_buffer=chunk,
                        **input_kwargs,
                    )
                    rate = candidate
                    print(f"[voice] 录音采样率: {rate}Hz")
                    break
                except Exception as exc:
                    stream_error = exc
                    print(f"[voice] 采样率 {candidate}Hz 打开失败: {exc}")
            if stream is None:
                _print_voice_backend_fallback_hint(rate)
                if stream_error is not None:
                    print(f"[voice] 语音触发不可用: {stream_error}")
                else:
                    print("[voice] 语音触发不可用: 没有可用的 PyAudio 输入采样率")
                return False
            while not self._stop.is_set():
                if _tts_playing.is_set() or time.time() < _voice_pause_until:
                    time.sleep(0.05)
                    continue
                data = stream.read(chunk, exception_on_overflow=False)
                rms = _compute_rms(data)
                now = time.time()
                if rms == 0:
                    if zero_input_since is None:
                        zero_input_since = now
                    elif (
                        not zero_input_hint_printed
                        and now - zero_input_since >= VOICE_ZERO_INPUT_WARN_SECONDS
                    ):
                        _print_zero_input_hint(str(dev_name), rate)
                        zero_input_hint_printed = True
                else:
                    zero_input_since = None
                if VOICE_DEBUG and now - last_rms_log >= 2.0:
                    print(f"[voice] RMS={rms} 阈值={VOICE_RMS_THRESHOLD}")
                    last_rms_log = now
                if rms < VOICE_RMS_THRESHOLD or now - last_trigger < VOICE_COOLDOWN_SECONDS:
                    continue
                last_trigger = now
                print(f"[voice] 触发录音: RMS={rms}，录音 {VOICE_COMMAND_SECONDS:.1f}s")
                frames = [data]
                frame_count = int(rate / chunk * VOICE_COMMAND_SECONDS)
                for _ in range(frame_count):
                    if self._stop.is_set():
                        break
                    frames.append(stream.read(chunk, exception_on_overflow=False))
                wav_path = _write_voice_wav(frames, rate)
                self._handle_voice_wav(wav_path)
            return True
        except Exception as exc:
            print(f"[voice] 语音触发不可用: {exc}")
            return False
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()

    def _recognize_voice(self, wav_path: Path) -> str:
        if not ENABLE_ASR:
            return ""
        if self._asr_unavailable_reason is not None:
            if not self._asr_unavailable_logged:
                print(f"[voice] ASR 已禁用: {self._asr_unavailable_reason}")
                self._asr_unavailable_logged = True
            return ""
        if not Path(SENSEVOICE_MODEL_DIR).is_dir():
            self._asr_unavailable_reason = f"模型目录不存在: {SENSEVOICE_MODEL_DIR}"
            if not self._asr_unavailable_logged:
                print(f"[voice] ASR 不可用: {self._asr_unavailable_reason}")
                self._asr_unavailable_logged = True
            return ""
        try:
            if self._asr_model is None:
                SenseVoiceSmall = _load_sensevoice_class(SENSEVOICE_MODEL_DIR)
                print("[voice] 加载 SenseVoice 模型...")
                self._asr_model = _create_sensevoice_model(SenseVoiceSmall, SENSEVOICE_MODEL_DIR)
                print("[voice] SenseVoice 模型就绪")
            result = _recognize_with_sensevoice(self._asr_model, wav_path)
            return extract_asr_text(result).strip()
        except ModuleNotFoundError as exc:
            self._asr_unavailable_reason = f"缺少模块: {exc}"
            if not self._asr_unavailable_logged:
                print(f"[voice] ASR 不可用: {self._asr_unavailable_reason}")
                self._asr_unavailable_logged = True
            return ""
        except Exception as exc:
            print(f"[voice] ASR 识别失败: {exc}")
            return ""


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _load_labels(path: str, default_labels: list[str]) -> list[str]:
    if not path:
        return default_labels
    label_path = Path(path)
    if not label_path.exists():
        return default_labels
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    return [label for label in labels if label] or default_labels


def _softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def _write_voice_wav(frames: list[bytes], rate: int) -> Path:
    wav_path = Path(tempfile.gettempdir()) / "k1_life_assistant_voice_cmd.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return wav_path


def _voice_temp_wav(name: str) -> Path:
    return Path(tempfile.gettempdir()) / name


def _read_wav_pcm(wav_path: Path) -> tuple[bytes, int]:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    if channels == 1 and sampwidth == 2:
        return data, rate
    if channels <= 1:
        return data, rate
    mono = bytearray()
    step = sampwidth * channels
    for offset in range(0, len(data) - step + 1, step):
        mono.extend(data[offset : offset + sampwidth])
    return bytes(mono), rate


def _run_arecord(
    wav_path: Path, seconds: float, rate: int, device: Optional[str] = None
) -> bool:
    if not _have("arecord"):
        print("[voice] arecord 不可用，请安装 alsa-utils")
        return False
    cmd = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(rate),
        "-t",
        "wav",
        "-d",
        str(max(1, int(math.ceil(seconds)))),
    ]
    if device:
        cmd.extend(["-D", device])
    cmd.append(str(wav_path))
    try:
        proc = subprocess.run(cmd, check=False, timeout=seconds + 3)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[voice] arecord 录音失败: {exc}")
        return False
    if proc.returncode != 0:
        print(f"[voice] arecord 录音失败，返回码 {proc.returncode}")
        return False
    return wav_path.exists() and wav_path.stat().st_size > 44


def _prepare_spacemit_asr_import() -> None:
    for raw_path in SPACEMIT_ASR_PATHS:
        path = os.path.expanduser(raw_path)
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)


def _load_sensevoice_class(model_dir: str):
    _prepare_spacemit_asr_import()
    import_errors: list[str] = []

    for module_name in (
        "spacemit_asr.models.sensevoice_bin",
        "llm_asr.models.sensevoice_bin",
    ):
        try:
            module = __import__(module_name, fromlist=["SenseVoiceSmall"])
            return module.SenseVoiceSmall
        except (ModuleNotFoundError, ImportError, AttributeError) as exc:
            import_errors.append(f"{module_name}: {exc}")

    try:
        from sensevoice import SenseVoiceSmall  # type: ignore

        return SenseVoiceSmall
    except ModuleNotFoundError as exc:
        import_errors.append(f"sensevoice: {exc}")

    for candidate in _sensevoice_model_py_candidates(model_dir):
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("k1_sensevoice_model", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.SenseVoiceSmall

    raise ModuleNotFoundError(
        "Cannot find SenseVoiceSmall. Tried: " + "; ".join(import_errors)
    )


def _create_sensevoice_model(SenseVoiceSmall, model_dir: str):
    if hasattr(SenseVoiceSmall, "from_pretrained"):
        return SenseVoiceSmall.from_pretrained(model_dir)
    for args, kwargs in (
        ((model_dir,), {"batch_size": 10, "quantize": True}),
        ((model_dir,), {"batch_size": 1, "quantize": True}),
        ((model_dir,), {}),
        ((), {}),
    ):
        try:
            return SenseVoiceSmall(*args, **kwargs)
        except TypeError:
            continue
    return SenseVoiceSmall(model_dir)


def _recognize_with_sensevoice(model, wav_path: Path):
    wav = str(wav_path)
    call_attempts = []
    for method_name in ("inference", "generate", "transcribe", "recognize"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        call_attempts.extend(
            [
                (method_name, method, (wav,), {"language": "zh"}),
                (method_name, method, (wav,), {"language": "zh", "use_itn": True}),
                (method_name, method, (wav,), {}),
            ]
        )
    if callable(model):
        call_attempts.extend(
            [
                ("__call__", model, (wav,), {"language": "zh"}),
                ("__call__", model, (wav,), {"language": "zh", "use_itn": True}),
                ("__call__", model, (wav,), {}),
            ]
        )

    errors: list[str] = []
    for name, func, args, kwargs in call_attempts:
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            errors.append(f"{name}: {exc}")
            continue
    raise AttributeError("SenseVoice model has no supported recognize method. " + "; ".join(errors))


def _sensevoice_model_py_candidates(model_dir: str) -> list[Path]:
    candidates = [Path(model_dir) / "model.py"]
    candidates.extend(
        [
            Path("/home/bianbu/spacemit-demo/examples/NLP/spacemit_asr/models/sensevoice_bin.py"),
            Path("~/spacemit-demo/examples/NLP/spacemit_asr/models/sensevoice_bin.py").expanduser(),
            Path("/opt/spacemit-asr/llm_asr/models/sensevoice_bin.py"),
            Path("/opt/spacemit-asr/cffi/model.py"),
        ]
    )
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _clean_asr_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    text = text.replace("\\n", " ").replace("\n", " ")
    return text.strip().strip("[]'\" ")


def _extract_asr_text_inner(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text", ""))
    if np is not None and isinstance(result, np.ndarray):
        if result.size == 0:
            return ""
        return _extract_asr_text_inner(result.reshape(-1)[0])
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text") or " ".join(first.get("preds", [])))
        return _extract_asr_text_inner(first)
    if isinstance(result, tuple) and result:
        return _extract_asr_text_inner(result[0])
    return ""


def extract_asr_text(result) -> str:
    return _clean_asr_text(_extract_asr_text_inner(result))


def route_voice_text(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if match_special_voice_reply(text) is not None:
        return "qa"
    if any(word in text for word in ("穿搭", "搭配", "衣服", "评价", "看看", "评分", "打分", "平分")):
        return "outfit"
    return None


def match_special_voice_reply(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if any(word in text for word in COLOR_SCORE_STANDARD_WORDS):
        return COLOR_SCORE_STANDARD_REPLY
    if not any(word in text for word in SPECIAL_LOCATION_WORDS):
        return None
    for _name, color_words, reply in SPECIAL_VOICE_REPLIES:
        if any(word in text for word in color_words):
            return reply
    return None


def is_exit_voice_text(text: str) -> bool:
    text = (text or "").strip()
    return any(word in text for word in ("退出", "再见", "拜拜", "结束", "关闭程序"))


def local_voice_reply(text: str) -> str:
    """无 Ollama 的本地规则问答，覆盖演示现场常见问题。"""
    text = (text or "").strip()
    if not text:
        return "我没有听清楚，请再说一遍。"
    special_reply = match_special_voice_reply(text)
    if special_reply is not None:
        return special_reply
    now = time.localtime()

    if any(word in text for word in ("你是谁", "叫什么", "介绍一下", "自我介绍")):
        return "我是 K1 视觉生活小助手。"
    if any(word in text for word in ("你能做什么", "有什么功能", "怎么用", "如何使用")):
        return "我可以做穿搭评分和简单问答。"
    if any(word in text for word in ("几点", "时间")):
        return time.strftime("现在是%H点%M分。", now)
    if any(word in text for word in ("今天几号", "日期", "星期几", "礼拜几")):
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return time.strftime(f"今天是%Y年%m月%d日，{weekdays[now.tm_wday]}。", now)
    if "天气" in text:
        return "我现在没有联网天气数据，不能确认实时天气。"
    if any(word in text for word in ("你好", "您好", "嗨")):
        return "你好，我在。你可以问问题，也可以让我看穿搭。"
    if any(word in text for word in ("谢谢", "感谢")):
        return "不客气。"
    if any(word in text for word in ("听不见", "没声音", "没有声音")):
        return "请确认没有设置 ENABLE_TTS=0，并检查播放设备和音箱连接。"
    if any(word in text for word in ("摄像头", "画面", "窗口")):
        return "摄像头用于穿搭评价。如果没有窗口，请在 K1 桌面终端运行，或检查 SHOW_WINDOW 设置。"
    return "这个问题我现在只能做简单本地回答。你可以问时间、功能，或者让我看穿搭。"


def save_capture(frame, prefix: str) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = CAPTURE_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def analyze_outfit_demo(frame, image_path: Path) -> AppResult:
    h, w = frame.shape[:2]
    x1, x2 = int(w * 0.22), int(w * 0.78)
    y1, y2 = int(h * 0.16), int(h * 0.92)
    body = frame[y1:y2, x1:x2]
    split = int(body.shape[0] * 0.47)
    top = _dominant_color_stats(body[:split, :])
    bottom = _dominant_color_stats(body[split:, :])

    score = 0
    reasons: list[str] = []
    if top.tone == bottom.tone and top.tone in ("warm", "cool"):
        score += 2
        reasons.append("全身色系统一，协调加分")
    elif "neutral" in (top.tone, bottom.tone):
        score += 1
        reasons.append("有中性色过渡，冲突较弱")
    elif {top.tone, bottom.tone} == {"warm", "cool"}:
        if top.is_large_bright and bottom.is_large_bright:
            score -= 2
            reasons.append("大面积亮冷色和亮暖色混穿，冲突扣分")
        else:
            score -= 1
            reasons.append("冷暖色混搭，需要控制面积")

    if top.brightness != bottom.brightness:
        score += 1
        reasons.append("上浅下深或上深下浅，有层次")
    else:
        score -= 1
        reasons.append("上下深浅接近，层次偏单调")

    if score >= 2:
        label = "搭配优秀"
        confidence = 0.78
    elif score >= 0:
        label = "搭配一般"
        confidence = 0.66
    else:
        label = "搭配需要调整"
        confidence = 0.62

    detail = (
        f"上半身偏{top.name}/{_brightness_cn(top.brightness)}，"
        f"下半身偏{bottom.name}/{_brightness_cn(bottom.brightness)}；"
        + "；".join(reasons)
    )
    return AppResult(
        title="穿搭智能评价",
        label=label,
        confidence=confidence,
        speech=_outfit_speech(label, detail),
        details=detail,
        mode="demo-color-rule",
        image_path=image_path,
    )


def _center_crop(frame, ratio: float):
    h, w = frame.shape[:2]
    cw, ch = int(w * ratio), int(h * ratio)
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    return frame[y1 : y1 + ch, x1 : x1 + cw]


def _dominant_color_stats(region) -> ColorStats:
    if region.size == 0:
        return ColorStats("灰色", "cool", "dark", 0.0, 0.0, 0.0)
    small = cv2.resize(region, (120, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    mask = (pixels[:, 1] > 25) & (pixels[:, 2] > 30)
    sample = pixels[mask] if int(mask.sum()) > 100 else pixels
    hist, _ = np.histogram(sample[:, 0], bins=36, range=(0, 180))
    idx = int(np.argmax(hist))
    hue_deg = (idx + 0.5) * 10.0
    saturation = float(sample[:, 1].mean())
    value = float(sample[:, 2].mean())
    name, tone = _classify_color(hue_deg, saturation, value)
    brightness = "light" if value >= 145 else "dark"
    return ColorStats(name, tone, brightness, hue_deg, saturation, value)


def _classify_color(hue_deg: float, saturation: float, value: float) -> tuple[str, str]:
    if value < 55:
        return "黑色", "cool"
    if saturation < 35:
        return "灰色", "cool"
    if hue_deg < 20 or hue_deg >= 340:
        return "红色", "warm"
    if 20 <= hue_deg < 70:
        return "橙黄色", "warm"
    if 300 <= hue_deg < 340:
        return "粉色", "warm"
    if 70 <= hue_deg < 170:
        return "绿色", "cool"
    if 170 <= hue_deg < 260:
        return "蓝色", "cool"
    return "冷色", "cool"


def _brightness_cn(value: str) -> str:
    return "浅色" if value == "light" else "深色"


def _outfit_speech(label: str, detail: str) -> str:
    score = _outfit_score(label)
    if label == "搭配优秀":
        return f"本次穿搭评分{score}分，配色协调，层次也比较清楚，整体效果很不错。"
    if label == "搭配一般":
        return f"本次穿搭评分{score}分，整体中规中矩，可以微调色彩或深浅层次。"
    return f"本次穿搭评分{score}分，颜色有些冲突，建议减少亮冷色和亮暖色的大面积混搭。"


def _outfit_score(label: str) -> int:
    if label == "搭配优秀":
        return 88
    if label == "搭配一般":
        return 72
    return 58


def draw_hud(frame, result: Optional[AppResult], message: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 170), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    lines = [
        "K1 视觉生活小助手 | O:穿搭评价  Q:退出",
        message,
    ]
    if result is not None:
        lines.extend(
            [
                f"{result.title}: {result.label}  置信度 {result.confidence:.2f}  模式 {result.mode}",
                result.details,
            ]
        )
    else:
        lines.append("待机中: 可说“帮我看看穿搭 / 蓝色的衣服在哪 / 颜色评分标准”，也可按 O。")

    draw_text_lines(frame, lines[:2], 18, 16, size=24)
    draw_text_lines(frame, lines[2:], 18, h - 132, size=24)
    return frame


def draw_text_lines(frame, lines: list[str], x: int, y: int, size: int = 24):
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        font_path = _find_chinese_font()
        if font_path:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, size)
            yy = y
            for line in lines:
                draw.text((x, yy), line, font=font, fill=(255, 255, 255))
                yy += size + 9
            frame[:, :] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return frame

    yy = y + size
    for line in lines:
        cv2.putText(
            frame,
            _ascii_overlay(line),
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        yy += size + 12
    return frame


def _find_chinese_font() -> Optional[str]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _ascii_overlay(text: str) -> str:
    replacements = {
        "搭配优秀": "OUTFIT GOOD",
        "搭配一般": "OUTFIT OK",
        "搭配需要调整": "OUTFIT ADJUST",
        "穿搭评价": "OUTFIT",
        "退出": "QUIT",
        "待机中": "STANDBY",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text.encode("ascii", errors="ignore").decode("ascii") or "K1 life assistant"


def _print_python_dependency_hint(module_name: str, apt_package: str, pip_package: str) -> None:
    python_cmd = shlex.quote(sys.executable)
    import_name = module_name.split("/")[-1]
    print(f"[x] 缺少 {module_name}: 当前 Python 无法导入该模块")
    print(f"    当前 Python: {sys.executable}")
    if ".venv" in sys.executable:
        print(
            "    如果 apt 显示系统包已安装但这里仍缺失，通常是当前 venv 看不到系统包。"
        )
        print(
            "    如果已用 apt 安装系统包，可执行: "
            f"{python_cmd} -m site"
        )
        print(
            "    若输出不含 /usr/lib/python3/dist-packages，重建或调整 venv 使其启用 "
            "system-site-packages。"
        )
        print(
            "    也可装入当前 venv: "
            f"{python_cmd} -m pip install {pip_package}"
        )
    else:
        print(f"    可尝试安装系统包: sudo apt install {apt_package}")
        print(f"    安装后验证: {python_cmd} -c \"import {import_name}\"")


def preflight() -> bool:
    print("=" * 60)
    print("K1 视觉生活小助手 - 自检")
    print("=" * 60)
    ok = True
    if cv2 is None:
        _print_python_dependency_hint("OpenCV/cv2", "python3-opencv", "opencv-python")
        ok = False
    else:
        print("[ok] OpenCV 可用")
    if np is None:
        _print_python_dependency_hint("numpy", "python3-numpy", "numpy")
        ok = False
    else:
        print("[ok] numpy 可用")
    if VOICE_BACKEND == "arecord":
        if _have("arecord"):
            device_hint = ARECORD_DEVICE or _default_arecord_device_from_aplay() or "default"
            print(f"[ok] 语音后端: arecord ALSA 直连，设备 {device_hint}")
        else:
            print("[warn] 语音后端: arecord 不可用，请安装 alsa-utils；仍可用键盘 O 演示")
    elif pyaudio is None:
        print("[warn] pyaudio 不可用，语音触发关闭；仍可用键盘 O 演示")
    else:
        print("[ok] 语音后端: pyaudio")
        _print_input_devices()
    if ENABLE_VOICE and ENABLE_VOICE_QA:
        print("[ok] 语音问答: 已启用本地规则回复，不依赖 Ollama")
    elif ENABLE_VOICE:
        print("[warn] 语音问答: ENABLE_VOICE_QA=0，仅识别穿搭口令")
    else:
        print("[warn] 语音输入: ENABLE_VOICE=0，已关闭")
    if CAMERA_INDEX_ENV not in (None, ""):
        if CAMERA_AUTO_FALLBACK:
            print(
                f"[ok] 摄像头索引: 优先 CAMERA_INDEX={CAMERA_INDEX}，失败后自动扫描"
            )
        else:
            print(f"[ok] 摄像头索引: 仅使用 CAMERA_INDEX={CAMERA_INDEX}")
    else:
        print(f"[ok] 摄像头索引: 自动扫描 /dev/video*，至少尝试 0..{CAMERA_SCAN_MAX}")
    if CAMERA_STARTUP:
        print("[ok] 摄像头模式: 启动即打开实时预览")
    else:
        print("[ok] 摄像头模式: 语音待机，穿搭评分时按需打开")
    if CAMERA_CLOSE_AFTER_ACTION:
        print("[ok] 摄像头释放: 视觉任务完成后自动关闭")
    else:
        print("[ok] 摄像头释放: 视觉任务完成后保持打开")
    print(f"[ok] 视觉任务预览: 触发后 {ACTION_PREVIEW_SECONDS:.1f}s 抓拍")
    print(
        f"[ok] 摄像头重连: 连续失败 {CAMERA_RECONNECT_FAILURES} 次后重试，间隔 {CAMERA_RECONNECT_SECONDS:.1f}s"
    )
    if SHOW_WINDOW and _display_available():
        print(f"[ok] 摄像头窗口: 将显示 OpenCV 窗口 `{WINDOW_NAME}`")
    elif SHOW_WINDOW:
        print("[warn] 图形显示不可用或无授权，SSH/串口终端里不会弹出摄像头窗口")
        print("       请在 K1 HDMI 桌面终端运行，或设置 SHOW_WINDOW=0 无窗口运行")
    else:
        print("[warn] SHOW_WINDOW=0，摄像头窗口已关闭")
    if ENABLE_ASR and Path(SENSEVOICE_MODEL_DIR).is_dir():
        print(f"[ok] SenseVoice 模型目录存在: {SENSEVOICE_MODEL_DIR}")
    else:
        print("[warn] SenseVoice 不可用时，只能键盘触发或音量兜底")
    if _spacemit_tts_available():
        print("[ok] TTS 合成: spacemit_tts")
    elif _have("espeak-ng") or _have("espeak"):
        fallback = "espeak-ng" if _have("espeak-ng") else "espeak"
        print(f"[warn] TTS 合成: {fallback}，未使用官方 spacemit_tts")
    else:
        print("[warn] 未找到 spacemit_tts/espeak-ng/espeak，TTS 只打印不播报")
    player = _pick_player()
    if player:
        if player == "aplay" and PLAY_DEVICE:
            print(f"[ok] 播放器: {player} -D {PLAY_DEVICE}")
            fallback_devices = [
                device for device, _card, _name in _discover_aplay_devices() if device != PLAY_DEVICE
            ]
            if fallback_devices:
                print(f"[ok] 播放设备兜底: {', '.join(fallback_devices)}")
        else:
            print(f"[ok] 播放器: {player}")
    else:
        print("[warn] 未找到 paplay/pw-play/aplay，TTS 可能无法播放")
    if MIC_INDEX is not None:
        print(f"[ok] 麦克风索引: MIC_INDEX={MIC_INDEX}")
    elif MIC_INDEX_ENV is not None:
        print(f"[warn] 麦克风索引: MIC_INDEX={MIC_INDEX_ENV!r} 无法解析，将使用系统默认输入")
    elif VOICE_BACKEND == "arecord":
        print("[ok] 麦克风索引: arecord 后端不使用 MIC_INDEX")
    else:
        print("[ok] 麦克风索引: 未指定，自动选择当前可用输入")
    if VOICE_SAMPLE_RATE:
        print(f"[ok] 录音采样率: VOICE_SAMPLE_RATE={VOICE_SAMPLE_RATE}")
        if VOICE_SAMPLE_RATE < 16000:
            print("[warn] 录音采样率过低，中文 ASR 识别可能明显变差，建议使用 44100 或 16000")
    elif VOICE_SAMPLE_RATE_ENV is not None:
        print(f"[warn] 录音采样率: VOICE_SAMPLE_RATE={VOICE_SAMPLE_RATE_ENV!r} 无法解析，将使用设备默认采样率")
    else:
        print("[ok] 录音采样率: 未指定，使用麦克风默认采样率")
    print(f"[ok] 语音触发阈值: VOICE_RMS_THRESHOLD={VOICE_RMS_THRESHOLD}")
    if VOICE_DEBUG:
        print("[ok] 语音调试: VOICE_DEBUG=1，将周期打印麦克风 RMS 和 ASR 文本")
    if not ENABLE_TTS:
        print("[warn] ENABLE_TTS=0，TTS 已关闭")
    elif not SAY_STARTUP:
        print("[warn] SAY_STARTUP=0，跳过启动播报")
    print("=" * 60)
    return ok


def process_action(action: str, frame, classifiers: ClassifierHub) -> AppResult:
    image_path = save_capture(frame, "outfit")
    result = classifiers.classify_outfit(frame, image_path)
    print(f"[result] {result.title} | {result.label} | {result.details} | {result.image_path}")
    tts_play(result.speech)
    return result


def print_camera_open_failure() -> None:
    if CAMERA_INDEX_ENV not in (None, ""):
        if CAMERA_AUTO_FALLBACK:
            print(f"[x] 摄像头打开失败: CAMERA_INDEX={CAMERA_INDEX} 及自动候选节点均不可用")
        else:
            print(f"[x] 摄像头打开失败: CAMERA_INDEX={CAMERA_INDEX}")
    else:
        print("[x] 摄像头打开失败: 自动扫描未找到可读图像流")
    print("    检查: ls /dev/video*")
    print("    检查: v4l2-ctl --list-devices")
    print("    检查占用: fuser -v /dev/video*")
    print("    注意: 同一个 USB Camera 常有两个 video 节点，其中一个可能不是图像流")


def main() -> int:
    if not preflight():
        return 1

    classifiers = ClassifierHub()
    voice = VoiceTrigger()
    voice.start()

    cap = None
    active_camera_index: Optional[int] = None
    window_enabled = False
    window_created = False
    result: Optional[AppResult] = None
    result_until = 0.0
    message = "待机中"
    if SAY_STARTUP:
        if CAMERA_STARTUP:
            tts_play("视觉生活小助手已启动。你可以问我问题，也可以让我看穿搭。")
        else:
            tts_play("语音助手已启动。你可以问我问题，需要穿搭评分时请说给我穿搭评分，也可以问蓝色的衣服在哪。")
    else:
        print("[tts] SAY_STARTUP=0，跳过启动播报")
    first_frame_logged = False
    read_failures = 0
    pending_action: Optional[str] = None
    pending_action_started_at: Optional[float] = None

    def ensure_camera() -> bool:
        nonlocal cap, active_camera_index, window_enabled, window_created, first_frame_logged
        if cap is not None:
            return True
        cap, active_camera_index = _open_camera()
        if cap is None or active_camera_index is None:
            print_camera_open_failure()
            return False
        first_frame_logged = False
        print(f"[ok] 摄像头已打开: CAMERA_INDEX={active_camera_index}")
        window_enabled = SHOW_WINDOW and _display_available()
        if window_enabled and not window_created:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                window_created = True
                print(f"[ok] 摄像头窗口已创建: {WINDOW_NAME}")
            except Exception as exc:
                window_enabled = False
                print(f"[warn] 摄像头窗口创建失败: {exc}")
                print("       仍会继续读取摄像头，但当前终端看不到画面")
        elif not window_enabled:
            print("[warn] 当前不会显示摄像头窗口；如需画面，请在 K1 桌面终端运行")
        return True

    def close_camera_after_action() -> None:
        nonlocal cap, active_camera_index, window_enabled, window_created, first_frame_logged, read_failures
        if not CAMERA_CLOSE_AFTER_ACTION or CAMERA_STARTUP:
            return
        if cap is not None:
            cap.release()
            cap = None
            active_camera_index = None
            first_frame_logged = False
            read_failures = 0
            print("[camera] 本次视觉任务完成，已释放摄像头")
        if window_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            window_created = False
            window_enabled = False

    def read_camera_frame():
        nonlocal cap, active_camera_index, first_frame_logged, read_failures
        if cap is None and not ensure_camera():
            return None
        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                print(f"[x] 摄像头读取异常: {exc}")
                ret, frame = False, None
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                if read_failures:
                    print("[ok] 摄像头读取恢复")
                    read_failures = 0
                if not first_frame_logged:
                    h, w = frame.shape[:2]
                    print(f"[ok] 摄像头读帧成功: {w}x{h}")
                    first_frame_logged = True
                return frame

            read_failures += 1
            print(f"[x] 摄像头读取失败 ({read_failures}/{CAMERA_RECONNECT_FAILURES})")
            if read_failures < CAMERA_RECONNECT_FAILURES:
                time.sleep(0.2)
                return None
            print(f"[camera] 连续读帧失败，{CAMERA_RECONNECT_SECONDS:.1f}s 后重新扫描摄像头")
            cap.release()
            cap = None
            active_camera_index = None
            time.sleep(CAMERA_RECONNECT_SECONDS)
            if not ensure_camera():
                print("[camera] 重连失败，将继续尝试")
                read_failures = CAMERA_RECONNECT_FAILURES - 1
                time.sleep(CAMERA_RECONNECT_SECONDS)
                return None
            read_failures = 0

    try:
        while True:
            event = voice.get_event()
            if event:
                action, text = event
                if is_exit_voice_text(text):
                    message = f"语音退出: {text}"
                    tts_play("好的，再见。")
                    break
                if action == "qa":
                    reply = local_voice_reply(text)
                    message = f"语音问答: {text}"
                    print(f"[qa] {text} -> {reply}")
                    tts_play(reply)
                else:
                    message = f"语音触发: {text}"
                    pending_action = action
                    pending_action_started_at = None
                    pending_text = text

            frame = None
            if pending_action is not None or cap is not None or CAMERA_STARTUP:
                frame = read_camera_frame()

            if pending_action is not None and frame is not None:
                now = time.time()
                if pending_action_started_at is None:
                    pending_action_started_at = now
                    print(
                        f"[camera] 视觉任务已触发，预览 {ACTION_PREVIEW_SECONDS:.1f}s 后抓拍"
                    )
                preview_elapsed = now - pending_action_started_at
                if preview_elapsed >= ACTION_PREVIEW_SECONDS:
                    result = process_action(pending_action, frame.copy(), classifiers)
                    result_until = time.time() + RESULT_HOLD_SECONDS
                    pending_action = None
                    pending_action_started_at = None
                    close_camera_after_action()
                else:
                    remain = max(0.0, ACTION_PREVIEW_SECONDS - preview_elapsed)
                    message = f"摄像头实时预览中，{remain:.1f}s 后抓拍评价"
            elif pending_action is not None and cap is None:
                tts_play("摄像头暂时打不开，无法完成视觉识别。")
                pending_action = None
                pending_action_started_at = None

            if time.time() > result_until:
                result = None

            if frame is not None:
                draw_hud(frame, result, message)
            key = 255
            if window_enabled and frame is not None:
                try:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as exc:
                    window_enabled = False
                    print(f"[warn] 摄像头窗口显示失败: {exc}")
                    print("       请确认是在 K1 HDMI 桌面终端运行，不是纯 SSH/串口终端")
            else:
                time.sleep(0.03)
            if key in (ord("q"), 27):
                break
            if key in (ord("o"), ord("O"), ord("f"), ord("F")):
                message = "键盘触发: 帮我看看穿搭"
                pending_action = "outfit"
                pending_action_started_at = None
            if key in (ord("t"), ord("T")):
                tts_play("语音播报测试。")
    except KeyboardInterrupt:
        print("\n[!] 用户中断")
    finally:
        voice.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
