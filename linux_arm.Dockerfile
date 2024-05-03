# docker build . -t maturin-aarm64 -f Dockerfile.build
# docker run -v .:/rottnest -w /rottnest maturin-aarm64 build --release --features py

FROM ghcr.io/pyo3/maturin
RUN yum update -y && yum install -y openssl-devel
