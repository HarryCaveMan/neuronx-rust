FROM --platform=linux/amd64 public.ecr.aws/amazonlinux/amazonlinux:2023-minimal

COPY al2023build.sh build.sh

RUN chmod +x build.sh && ./build.sh

ENV PATH="/opt/aws/neuron/bin:${PATH}"
ENV LIBNRT_LIB_PATH=/opt/aws/neuron/lib
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTFLAGS=-Awarnings
ENV RUSTDOCFLAGS=-Awarnings

ENTRYPOINT ["/bin/bash"]