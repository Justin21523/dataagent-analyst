#!/usr/bin/env bash

set -euo pipefail

ECHARTS_VERSION="${ECHARTS_VERSION:-6.1.0}"
VENDOR_DIR="frontend/vendor"

ECHARTS_URL="https://cdn.jsdelivr.net/npm/echarts@${ECHARTS_VERSION}/dist/echarts.min.js"
LICENSE_URL="https://raw.githubusercontent.com/apache/echarts/${ECHARTS_VERSION}/LICENSE"
NOTICE_URL="https://raw.githubusercontent.com/apache/echarts/${ECHARTS_VERSION}/NOTICE"

mkdir -p "${VENDOR_DIR}"

echo "==> Downloading Apache ECharts ${ECHARTS_VERSION}"

curl \
  --fail \
  --location \
  --retry 3 \
  --retry-delay 2 \
  "${ECHARTS_URL}" \
  --output "${VENDOR_DIR}/echarts.min.js"

curl \
  --fail \
  --location \
  --retry 3 \
  "${LICENSE_URL}" \
  --output "${VENDOR_DIR}/ECHARTS_LICENSE"

curl \
  --fail \
  --location \
  --retry 3 \
  "${NOTICE_URL}" \
  --output "${VENDOR_DIR}/ECHARTS_NOTICE"

echo "${ECHARTS_VERSION}" > "${VENDOR_DIR}/ECHARTS_VERSION"

file_size="$(
  wc -c < "${VENDOR_DIR}/echarts.min.js"
)"

if [[ "${file_size}" -lt 500000 ]]; then
  echo "Downloaded ECharts file looks too small: ${file_size} bytes"
  exit 1
fi

echo "==> ECharts vendor installation completed"
echo "    version=${ECHARTS_VERSION}"
echo "    size=${file_size} bytes"
