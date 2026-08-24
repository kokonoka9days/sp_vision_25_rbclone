#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$(cd -- "${script_dir}/../models" && pwd)"
onnx_path="${model_dir}/model-0624.onnx"
engine_path="${1:-${model_dir}/model-0624-fp16.engine}"
expected_sha256="e6e6bddf1f793ece1b9b351b8db061dbb026d32b8a6d71cb39a8fd21ce9cb917"

trtexec_bin="${TRTEXEC:-}"
if [[ -z "${trtexec_bin}" ]] && command -v trtexec >/dev/null 2>&1; then
  trtexec_bin="$(command -v trtexec)"
fi
if [[ -z "${trtexec_bin}" && -n "${TENSORRT_ROOT:-}" && -x "${TENSORRT_ROOT}/bin/trtexec" ]]; then
  trtexec_bin="${TENSORRT_ROOT}/bin/trtexec"
fi
if [[ -z "${trtexec_bin}" ]]; then
  echo "trtexec is not available; install the TensorRT command-line tools." >&2
  exit 1
fi

actual_sha256="$(sha256sum "${onnx_path}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "model-0624.onnx checksum mismatch: ${actual_sha256}" >&2
  exit 1
fi

"${trtexec_bin}" \
  --onnx="${onnx_path}" \
  --saveEngine="${engine_path}" \
  --fp16 \
  --skipInference

echo "TensorRT rune engine written to ${engine_path}"
