package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"ner-backend/pkg/api"
	"net/http"
	"regexp"
	"sort"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/gorilla/schema"
)

type codedError struct {
	err  error
	code int
}

func (e *codedError) Error() string {
	return e.err.Error()
}

func (e *codedError) Unwrap() error {
	return e.err
}

func CodedError(code int, err error) error {
	return &codedError{err: err, code: code}
}

func CodedErrorf(code int, format string, args ...any) error {
	return &codedError{err: fmt.Errorf(format, args...), code: code}
}

func ParseRequest[T any](r *http.Request) (T, error) {
	var data T
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		slog.Error("error parsing request body", "error", err)
		return data, CodedErrorf(http.StatusBadRequest, "unable to parse request body")
	}
	return data, nil
}

func ParseRequestQueryParams[T any](r *http.Request) (T, error) {
	var data T
	if err := r.ParseForm(); err != nil {
		slog.Error("error parsing form", "error", err)
		return data, CodedErrorf(http.StatusBadRequest, "unable to parse request query params")
	}

	err := schema.NewDecoder().Decode(&data, r.Form)
	if err != nil {
		slog.Error("error decoding query params", "error", err)
		return data, CodedErrorf(http.StatusBadRequest, "unable to parse request query params")
	}

	return data, nil
}

func RestHandler(handler func(r *http.Request) (any, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		res, err := handler(r)
		if err != nil {
			var cerr *codedError
			if errors.As(err, &cerr) {
				http.Error(w, err.Error(), cerr.code)
				if cerr.code == http.StatusInternalServerError {
					slog.Error("internal server error received in endpoint", "error", err)
				}
			} else {
				slog.Error("recieved non coded error from endpoint", "error", err)
				http.Error(w, err.Error(), http.StatusInternalServerError)

			}
			return
		}

		if res == nil {
			res = struct{}{}
		}

		WriteJsonResponse(w, res)
	}
}

type StreamResponse func(yield func(any, error) bool)

type StreamMessage struct {
	Data  interface{}
	Error string
	Code  int
}

func RestStreamHandler(handler func(r *http.Request) (StreamResponse, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		stream, err := handler(r)
		if err != nil {
			var cerr *codedError
			if errors.As(err, &cerr) {
				http.Error(w, err.Error(), cerr.code)
				if cerr.code == http.StatusInternalServerError {
					slog.Error("internal server error received in endpoint", "error", err)
				}
			} else {
				slog.Error("recieved non coded error from endpoint", "error", err)
				http.Error(w, err.Error(), http.StatusInternalServerError)

			}
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("response writer does not support flushing")
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		for data, err := range stream {
			var msg StreamMessage
			if err != nil {
				var cerr *codedError
				if errors.As(err, &cerr) {
					msg = StreamMessage{
						Error: err.Error(),
						Code:  cerr.code,
					}
					if cerr.code == http.StatusInternalServerError {
						slog.Error("internal server error received in endpoint", "error", err)
					}
				} else {
					msg = StreamMessage{
						Error: err.Error(),
						Code:  http.StatusInternalServerError,
					}
					slog.Error("received non coded error from endpoint", "error", err)
				}
			} else {
				msg = StreamMessage{
					Data: data,
					Code: http.StatusOK,
				}
			}

			if writeErr := json.NewEncoder(w).Encode(msg); writeErr != nil {
				slog.Error("error writing json response", "error", writeErr)
				return
			}

			flusher.Flush()
		}
	}
}

func WriteJsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	err := json.NewEncoder(w).Encode(data)
	if err != nil {
		slog.Error("error serializing response body", "error", err)
		http.Error(w, fmt.Sprintf("error serializing response body: %v", err), http.StatusInternalServerError)
	}
}

func URLParamUUID(r *http.Request, key string) (uuid.UUID, error) {
	param := chi.URLParam(r, key)

	if len(param) == 0 {
		return uuid.Nil, CodedErrorf(http.StatusBadRequest, "missing {%v} url parameter", key)
	}

	id, err := uuid.Parse(param)
	if err != nil {
		return uuid.Nil, CodedErrorf(http.StatusBadRequest, "invalid uuid '%v' url parameter provided: %w", key, err)
	}

	return id, nil
}

func validateName(name string) error {
	// Allow only alphanumeric characters, underscores, and hyphens
	matched, err := regexp.MatchString("^[\\w-]+$", name)
	if err != nil {
		return CodedError(http.StatusInternalServerError, fmt.Errorf("error validating report name: %w", err))
	}

	if !matched {
		return CodedErrorf(http.StatusBadRequest, "invalid report name '%s' provided: only alphanumeric characters, underscores, and hyphens are allowed", name)
	}

	return nil
}

func MedianWordCount(samples []api.Sample) int {
	if len(samples) == 0 {
		return 0
	}

	wordCounts := make([]int, len(samples))
	for i, sample := range samples {
		wordCounts[i] = len(sample.Tokens)
	}

	sort.Ints(wordCounts)

	mid := len(wordCounts) / 2
	if len(wordCounts)%2 == 0 {
		return (wordCounts[mid-1] + wordCounts[mid]) / 2
	}
	return wordCounts[mid]
}

func AutoTuneK(samples []api.Sample, baseK int, alpha float64) int {
	count := MedianWordCount(samples)
	if count <= 0 {
		return baseK
	}
	k := float64(baseK) * math.Log(1+alpha/float64(count))
	if k < 2 {
		return 2
	}
	return int(k)
}
