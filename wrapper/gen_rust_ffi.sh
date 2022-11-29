#!/bin/bash

~/rust-bindgen/target/debug/bindgen \
  --whitelist-type="^CU.*" \
  --whitelist-type="^cuuint64_t" \
  --whitelist-type="^cudaError_enum" \
  --whitelist-type="^cu.*Complex$" \
  --whitelist-type="^cuda.*" \
  --whitelist-type="^libraryPropertyType.*" \
  --whitelist-var="^CU.*" \
  --whitelist-function="^cu.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  cuda-headers.h -- -I/usr/local/cuda-11.8/targets/x86_64-linux/include \
  > src/driver-ffi.rs