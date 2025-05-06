# --- Build Stage ---
FROM golang:1.24 as builder

WORKDIR /app

# Copy module files and download dependencies first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -v -o api    ./cmd/api
RUN CGO_ENABLED=1 GOOS=linux go build -v -o worker ./cmd/worker

# --- Final Stage: lightweight runtime with Python support ---
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# TODO(pratik): remove this when the thirdai wheel is available with APIs in PYPI
COPY --from=builder /app/plugin/plugin-python/requirements.txt ./plugin/plugin-python/
COPY --from=builder /app/cmd/thirdai-0.9.29-*.whl      ./

RUN pip install --no-cache-dir \
    -r plugin/plugin-python/requirements.txt 


RUN pip install --no-cache-dir \
    thirdai-0.9.29-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY --from=builder /app/api          /app/api
COPY --from=builder /app/worker       /app/worker
COPY --from=builder /app/entrypoint.sh /app/entrypoint.sh

COPY --from=builder /app/plugin/plugin-python /app/plugin/plugin-python

RUN chmod +x /app/entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--worker"]