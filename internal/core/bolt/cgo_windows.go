//go:build windows && cgo
// +build windows,cgo

package bolt

import "C"

// This file exists to satisfy cgo requirements on Windows without including C++ files