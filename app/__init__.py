"""RAG ablation app package.

HF 直连在国内实测超时，且镜像不支持 Xet CAS（会 401）。这里给默认值，可用环境变量覆盖。
两者都是模型下载必需，写在这里保证任何入口（run/sweep）都能复用同一份缓存。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
