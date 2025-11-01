#!/usr/bin/env bash
set -euo pipefail

_have() { command -v "$1" >/dev/null 2>&1; }

echo "== NVIDIA driver install detection =="

driver_ver="$(_have nvidia-smi && nvidia-smi --query-gpu=driver_version \
  --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "Driver reported by nvidia-smi: ${driver_ver:-unknown}"

owner="unknown"
if _have dpkg && _have nvidia-smi && dpkg -S "$(command -v nvidia-smi)" >/dev/null 2>&1; then
  owner_pkg="$(dpkg -S "$(command -v nvidia-smi)" | cut -d: -f1 | head -n1)"
  owner="apt (${owner_pkg})"
elif [ -x /usr/bin/nvidia-uninstall ] || [ -f /var/log/nvidia-installer.log ]; then
  owner=".run installer"
fi
echo "User-space tool ownership: ${owner}"

echo
echo "== DKMS status =="
dkms status | grep -i nvidia || echo "No DKMS records for nvidia"

echo
echo "== Installed apt packages (nvidia, libnvidia, cuda-keyring) =="
apt list --installed 2>/dev/null | grep -E '^(nvidia|libnvidia|cuda-keyring)' || true

echo
echo "== Repository origins for the driver meta-package =="
driver_pkg="$(dpkg -l | awk '/^ii/ && $2 ~ /^nvidia-driver-[0-9]+$/ {print $2; exit}')"
if [ -n "${driver_pkg:-}" ]; then
  apt-cache policy "$driver_pkg"
else
  echo "No nvidia-driver-XXX meta-package registered."
fi

echo
echo "== APT sources mentioning NVIDIA/CUDA/graphics-drivers PPA =="
grep -HnE 'nvidia|cuda|graphics-drivers' /etc/apt/sources.list \
  /etc/apt/sources.list.d/*.list 2>/dev/null || true

echo
echo "== Kernel module path hint =="
modinfo nvidia 2>/dev/null | awk '/^filename:/ {print $0; exit}'

echo
echo "== Classification =="
if _have dpkg && _have nvidia-smi && dpkg -S "$(command -v nvidia-smi)" >/dev/null 2>&1; then
  if [ -n "${driver_pkg:-}" ]; then
    origin="$(apt-cache policy "$driver_pkg")"
    if grep -qi 'ppa.launchpad.net/graphics-drivers' <<<"$origin"; then
      echo "APT-managed via Graphics Drivers PPA."
    elif grep -qi 'download.nvidia.com' <<<"$origin"; then
      echo "APT-managed via NVIDIA official repository."
    else
      echo "APT-managed via Ubuntu archive."
    fi
  else
    echo "APT-managed (tools owned by dpkg), driver meta-package not detected."
  fi
elif [ -x /usr/bin/nvidia-uninstall ] || [ -f /var/log/nvidia-installer.log ]; then
  echo "Installed via NVIDIA .run installer."
else
  echo "Indeterminate. Likely APT-managed but missing meta-package; inspect outputs above."
fi
