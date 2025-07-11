//go:build windows && cgo
// +build windows,cgo

package bolt

// This file exists to satisfy cgo requirements on Windows without including C++ files
import "C"