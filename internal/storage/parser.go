package storage

import (
	"fmt"
	"io"
	"log/slog"
	"path/filepath"
	"strings"

	"github.com/gen2brain/go-fitz"
)

type Parser interface {
	Parse(object string, data io.Reader) chan Chunk
}

type DefaultParser struct {
	maxChunkSize int
}

const (
	defaultMaxChunkSize = 512 * 1024 * 1024 // 512 MB
	queueBufferSize     = 4
)

func NewDefaultParser() *DefaultParser {
	return &DefaultParser{maxChunkSize: defaultMaxChunkSize}
}

func (parser *DefaultParser) Parse(object string, data io.Reader) chan Chunk {
	output := make(chan Chunk, queueBufferSize)

	ext := filepath.Ext(object)

	go func() {
		defer close(output)

		switch ext {
		case ".pdf":
			parser.parsePdf(object, data, output)
		case ".txt", ".csv", ".html", ".json", ".xml":
			parser.parsePlaintext(object, data, output)
		default:
			slog.Warn("unsupported file type", "object", object)
		}
	}()

	return output
}

func (parser *DefaultParser) parsePdf(object string, data io.Reader, output chan Chunk) {
	document := make([]byte, parser.maxChunkSize)

	n, err := io.ReadFull(data, document)
	if err == nil {
		// if the error is nil then the end of the stream was not reached, thus we cannot parse the pdf.
		output <- Chunk{Error: fmt.Errorf("pdf is too large for parsing")}
		return
	} else if err != io.EOF && err != io.ErrUnexpectedEOF {
		output <- Chunk{Error: err}
		return
	}

	document = document[:n]

	pdf, err := fitz.NewFromMemory(document)
	if err != nil {
		output <- Chunk{Error: err}
		return
	}
	defer pdf.Close()

	pages := make([]string, 0, pdf.NumPage())

	for i := 0; i < pdf.NumPage(); i++ {
		pageText, err := pdf.Text(i)
		if err != nil {
			output <- Chunk{Error: err}
			return
		}
		pages = append(pages, pageText)
	}

	output <- Chunk{
		Text:    strings.Join(pages, "\n\n"),
		Offset:  0,
		Error:   nil,
		RawSize: int64(n),
	}
}

func (parser *DefaultParser) parsePlaintext(filename string, data io.Reader, output chan Chunk) {
	offset := 0
	for {
		chunk := make([]byte, parser.maxChunkSize)

		n, err := io.ReadFull(data, chunk)
		isEnd := false
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			err = nil
			isEnd = true
		}

		output <- Chunk{
			Text:    string(chunk[:n]),
			Offset:  offset,
			Error:   err,
			RawSize: int64(n),
		}
		offset += n

		if isEnd {
			return
		}
	}
}
