#!/bin/bash

# Set minimum deployment target to match current macOS version
export CGO_CFLAGS="-mmacosx-version-min=15.0"
export CGO_LDFLAGS="-mmacosx-version-min=15.0"

# Build the project
go build ./cmd/local/main.go 