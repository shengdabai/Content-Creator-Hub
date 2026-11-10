#!/usr/bin/env python3
"""
YouTube Creator Agent - 自动创作视频并发布到 YouTube

工具: read_file, write_file, generate_assets, compose_video, upload_youtube
"""

import os
import subprocess
import json
import sys
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# Ensure edge-tts, ffmpeg etc. are in PATH.
# Common install locations plus any extra entries the user configures
# via the PATH_EXTRA env var (colon-separated). No machine-specific paths
# are hardcoded — set PATH_EXTRA in your .env if your tools live elsewhere.
_path_extra = [p for p in os.getenv("PATH_EXTRA", "").split(":") if p]
for p in _path_extra + ["/opt/homebrew/bin", "/usr/local/bin"]:
    if p and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":" + p

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ.get("MODEL_ID", "claude-opus-4-5")
WORKDIR = Path.cwd()
OUTPUT_DIR = WORKDIR / "output"
if OUTPUT_DIR.is_symlink() or (OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir()):
    raise RuntimeError("Unsafe output directory")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_TEXT_BYTES = 100_000

SYSTEM = f"""你是一个专业的 YouTube 视频创作 agent，工作目录: {WORKDIR}

你的能力:
1. 根据用户给定的主题，创作完整的视频脚本（标题、描述、逐段文案）
2. 生成视频素材（配音、字幕、背景图片）
3. 用 ffmpeg 合成视频
4. 上传到 YouTube

工作流程:
- 收到主题后，先用 write_file 写脚本到 output/script.json
- 然后用 generate_assets 生成素材
- 再用 compose_video 合成视频
- 最后用 upload_youtube 上传

脚本 JSON 格式:
{{
  "title": "视频标题",
  "description": "YouTube 描述",
  "tags": ["标签1", "标签2"],
  "language": "zh-CN",
  "segments": [
    {{"text": "这一段的旁白文案", "duration": 10}},
    ...
  ]
}}

规则:
- 每段 segment 控制在 5-15 秒
- 总时长建议 1-5 分钟
- 文案要口语化、有节奏感
- 标题要有吸引力但不标题党
"""

TOOLS = [
    {
        "name": "read_file",
        "description": "读取文件内容",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "写入文件。用于保存脚本、字幕等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "generate_assets",
        "description": "根据 script.json 生成视频素材：edge-tts 配音 + SRT 字幕 + 纯色背景图。输入 script_path 指向脚本 JSON 文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "脚本 JSON 文件路径，如 output/script.json",
                }
            },
            "required": ["script_path"],
        },
    },
    {
        "name": "compose_video",
        "description": "用 ffmpeg 将配音、字幕、背景合成为最终 MP4 视频。",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "输出视频路径，如 output/final.mp4",
                }
            },
            "required": ["output_path"],
        },
    },
    {
        "name": "upload_youtube",
        "description": "将视频上传到 YouTube。需要先完成 OAuth 授权。",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "privacy": {"type": "string", "enum": ["public", "unlisted", "private"]},
            },
            "required": ["video_path", "title", "description"],
        },
    },
]


def run_bash(command: str) -> str:
    """Compatibility stub for old tool calls; never execute model commands."""
    return "Error: shell tool is disabled"


def _output_path(path: str) -> Path:
    candidate = WORKDIR / path
    try:
        relative = candidate.relative_to(OUTPUT_DIR)
    except ValueError:
        raise ValueError("Files must be inside output/") from None
    if not relative.parts:
        raise ValueError("A filename is required")
    current = OUTPUT_DIR
    for part in relative.parts:
        lower = part.lower()
        if (part.startswith('.') or lower.startswith(('credentials.', 'secrets.', 'id_rsa'))
                or lower in {'auth.json', 'youtube_token.json', 'youtube_credentials.json'}
                or lower.endswith(('.key', '.pem'))):
            raise ValueError("Hidden, parent, and credential paths are not allowed")
        current = current / part
        if current.is_symlink() or (current.exists() and current.stat().st_nlink > 1
                                    and not current.is_dir()):
            raise ValueError("Linked files are not allowed")
        if current.exists() and not (current.is_dir() or current.is_file()):
            raise ValueError("Only regular files and directories are allowed")
    if OUTPUT_DIR.is_symlink():
        raise ValueError("Linked output directory is not allowed")
    return candidate


def run_read_file(path: str) -> str:
    try:
        with _output_path(path).open('rb') as stream:
            data = stream.read(MAX_TEXT_BYTES + 1)
        if len(data) > MAX_TEXT_BYTES:
            return "Error: text file is too large"
        return data.decode('utf-8')
    except Exception as e:
        return f"Error: {e}"


def run_write_file(path: str, content: str) -> str:
    try:
        if len(content.encode('utf-8')) > MAX_TEXT_BYTES:
            return "Error: text file is too large"
        p = _output_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"已写入 {len(content)} 字节到 {path}"
    except Exception as e:
        return f"Error: {e}"


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_generate_assets(script_path: str) -> str:
    try:
        script = json.loads(run_read_file(script_path))
    except Exception as e:
        return f"Error: 读取脚本失败 - {e}"

    segments = script.get("segments", []) if isinstance(script, dict) else []
    if not isinstance(segments, list) or not segments or len(segments) > 100:
        return "Error: 脚本中没有 segments"
    if any(not isinstance(seg, dict) or not isinstance(seg.get('text'), str) for seg in segments):
        return "Error: segments must contain text strings"

    lang = script.get("language", "zh-CN")
    voice_map = {
        "zh-CN": "zh-CN-YunxiNeural",
        "zh-TW": "zh-TW-HsiaoChenNeural",
        "en": "en-US-GuyNeural",
        "ja": "ja-JP-KeitaNeural",
    }
    voice = voice_map.get(lang, "zh-CN-YunxiNeural")

    audio_dir = _output_path("output/audio")
    audio_dir.mkdir(exist_ok=True)

    check = subprocess.run(["edge-tts", "--version"], capture_output=True, text=True, timeout=15)
    if check.returncode != 0:
        return "Error: edge-tts 未安装。请运行: pip install edge-tts"

    results = []
    srt_lines = []
    time_offset = 0.0
    files = []

    for i, seg in enumerate(segments):
        text = seg["text"]
        audio_file = _output_path(f"output/audio/seg_{i:03d}.mp3")
        cmd = ['edge-tts', '--voice', voice, '--text', text, '--write-media', str(audio_file)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            results.append(f"  段落 {i}: TTS 失败 - {r.stderr[:100]}")
            return "Error: TTS failed; assets are incomplete"
        files.append(audio_file)

        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(audio_file)]
        dur_r = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=30)
        try:
            duration = float(dur_r.stdout.strip())
        except ValueError:
            duration = seg.get("duration", 10)

        start_ts = format_srt_time(time_offset)
        end_ts = format_srt_time(time_offset + duration)
        srt_lines.extend([f"{i + 1}", f"{start_ts} --> {end_ts}", text, ""])
        time_offset += duration
        results.append(f"  段落 {i}: {duration:.1f}s ✓")

    srt_path = _output_path("output/subtitles.srt")
    srt_path.write_text("\n".join(srt_lines))

    if files:
        file_list = _output_path("output/audio_list.txt")
        file_list.write_text("\n".join(f"file 'audio/{f.name}'" for f in files))
        narration = _output_path("output/narration.mp3")
        merge_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '1', '-i', str(file_list), '-c', 'copy', str(narration)]
        subprocess.run(merge_cmd, check=True, capture_output=True, timeout=120)

    background = _output_path("output/background.png")
    bg_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=0x1a1a2e:s=1920x1080:d=1', '-frames:v', '1', str(background)]
    subprocess.run(bg_cmd, check=True, capture_output=True, timeout=30)

    return (f"素材生成完成:\n  总时长: {time_offset:.1f}s\n"
            f"  音频: output/narration.mp3\n  字幕: output/subtitles.srt\n"
            f"  背景: output/background.png\n" + "\n".join(results))


def run_compose_video(output_path: str) -> str:
    narration = _output_path("output/narration.mp3")
    background = _output_path("output/background.png")

    if not narration.exists():
        return "Error: narration.mp3 不存在，请先 generate_assets"

    out = _output_path(output_path)
    if out.suffix.lower() != '.mp4':
        return "Error: output must be an MP4 file"
    out.parent.mkdir(parents=True, exist_ok=True)

    dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(narration)]
    dur_r = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=30)
    try:
        duration = float(dur_r.stdout.strip())
    except ValueError:
        duration = 60

    cmd = ['ffmpeg', '-y', '-loop', '1', '-i', str(background), '-i', str(narration),
           '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k',
           '-pix_fmt', 'yuv420p', '-shortest', '-t', str(duration), str(out)]
    # Note: SRT subtitles saved separately for YouTube upload
    r = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return f"Error: ffmpeg 合成失败\n{r.stderr[:500]}"

    size_mb = out.stat().st_size / (1024 * 1024)
    return f"视频合成完成: {output_path} ({size_mb:.1f} MB, {duration:.0f}s)"


def run_upload_youtube(video_path: str, title: str, description: str,
                       tags: list = None, privacy: str = "private") -> str:
    video_file = _output_path(video_path)
    if not video_file.is_file() or video_file.suffix.lower() != '.mp4':
        return "Error: upload requires an existing MP4 in output/"
    if privacy not in {'private', 'unlisted', 'public'}:
        return "Error: invalid privacy setting"
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "Error: upload requires approval at an interactive terminal"
    print(repr({'video': video_path, 'title': title, 'description': description,
                'tags': tags or [], 'privacy': privacy}))
    try:
        if input('Type PUBLISH to upload this video: ').strip() != 'PUBLISH':
            return "Upload cancelled"
    except (EOFError, KeyboardInterrupt):
        return "Upload cancelled"
    creds_file = WORKDIR / "youtube_credentials.json"
    token_file = WORKDIR / "youtube_token.json"

    if not creds_file.exists():
        client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
        client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")
        if not client_id or client_id == "你的CLIENT_ID":
            return (
                "Error: YouTube API 未配置。请完成以下步骤:\n"
                "1. 访问 https://console.cloud.google.com\n"
                "2. 创建项目 → 启用 YouTube Data API v3\n"
                "3. 创建 OAuth 2.0 Client ID (桌面应用类型)\n"
                "4. 下载 JSON 凭证保存为 youtube_credentials.json\n"
                "   或在 .env 中填写 YOUTUBE_CLIENT_ID 和 YOUTUBE_CLIENT_SECRET\n"
                "5. 首次运行会打开浏览器完成授权"
            )
        creds_data = {
            "installed": {
                "client_id": client_id, "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        creds_file.write_text(json.dumps(creds_data, indent=2))

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        return "Error: 缺少依赖。请运行: pip install google-auth-oauthlib google-api-python-client"

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {"title": title, "description": description,
                     "tags": tags or [], "categoryId": "22"},
        "status": {"privacyStatus": privacy},
    }
    media = MediaFileUpload(str(video_file), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response["id"]
    return f"上传成功!\nURL: https://youtube.com/watch?v={video_id}\n隐私: {privacy}"


def execute_tool(name: str, args: dict) -> str:
    dispatch = {
        "read_file": lambda: run_read_file(args["path"]),
        "write_file": lambda: run_write_file(args["path"], args["content"]),
        "generate_assets": lambda: run_generate_assets(args["script_path"]),
        "compose_video": lambda: run_compose_video(args["output_path"]),
        "upload_youtube": lambda: run_upload_youtube(**args),
    }
    try:
        return dispatch.get(name, lambda: f"Unknown tool: {name}")()
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        return f"Error: {type(exc).__name__}: video tool failed"


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m> {block.name}\033[0m: {str(block.input)[:120]}")
                output = execute_tool(block.name, block.input)
                print(output[:300])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("YouTube Creator Agent")
    print(f"工作目录: {WORKDIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("输入主题开始创作，输入 q 退出\n")

    history = []
    while True:
        try:
            query = input("\033[36myoutube >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]["content"]
        if isinstance(last, list):
            for block in last:
                if hasattr(block, "text"):
                    print(block.text)
        print()
