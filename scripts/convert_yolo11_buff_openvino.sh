#!/usr/bin/env bash
set -euo pipefail

# Convert the buff ONNX model to an OpenVINO IR with FP16-compressed weights.
# Run from the repository root, or pass an alternate repository root as $1.
repo_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
input_model="${repo_root}/assets/yolo11_buff_12point_v3.onnx"
output_model="${repo_root}/assets/yolo11_buff_12point_v3_fp16"

if ! command -v ovc >/dev/null 2>&1; then
  echo "error: OpenVINO Model Optimizer (ovc) was not found in PATH" >&2
  exit 1
fi
if [[ ! -f "${input_model}" ]]; then
  echo "error: input model does not exist: ${input_model}" >&2
  exit 1
fi

ovc "${input_model}" \
  --output_model "${output_model}" \
  --compress_to_fp16=True

echo "OpenVINO IR written to ${output_model}.xml and ${output_model}.bin"
