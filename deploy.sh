#!/usr/bin/env bash
set -euo pipefail

if ! command -v modal >/dev/null 2>&1; then
  echo "Modal CLI not found. Install via: pip install modal"
  exit 1
fi

echo "Deploying spatialGPT LLaVA-3D to Modal..."
modal deploy modal_app.py

echo ""
echo "Deployment requested. To view logs in real time, run:"
echo "  modal logs -f spatialgpt-llava3d.web"
echo ""
echo "Once the build finishes, the public URL will be printed in the deploy output. You can also list apps with:"
echo "  modal app list"
