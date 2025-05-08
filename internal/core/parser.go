package core

import (
	"fmt"
	"io"
	"log/slog"
	"path/filepath"
	"strings"

	"github.com/gen2brain/go-fitz"
)

type ParsedChunk struct {
	Text   string
	Offset int
	Error  error
}

type Parser interface {
	Parse(object string, data io.Reader) chan ParsedChunk
}

type DefaultParser struct {
	maxChunkSize int
}

const (
	defaultMaxChunkSize = 512 * 1024 * 1024 // 2 MB
	queueBufferSize     = 4
)

func NewDefaultParser() *DefaultParser {
	return &DefaultParser{maxChunkSize: defaultMaxChunkSize}
}

func (parser *DefaultParser) Parse(object string, data io.Reader) chan ParsedChunk {
	output := make(chan ParsedChunk, queueBufferSize)

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

func (parser *DefaultParser) parsePdf(object string, data io.Reader, output chan ParsedChunk) {
	document := make([]byte, parser.maxChunkSize)

	n, err := io.ReadFull(data, document)
	if err == nil {
		// if the error is nil then the end of the stream was not reached, thus we cannot parse the pdf.
		output <- ParsedChunk{Error: fmt.Errorf("pdf is too large for parsing")}
		return
	} else if err != io.EOF && err != io.ErrUnexpectedEOF {
		output <- ParsedChunk{Error: err}
		return
	}

	document = document[:n]

	pdf, err := fitz.NewFromMemory(document)
	if err != nil {
		output <- ParsedChunk{Error: err}
		return
	}
	defer pdf.Close()

	pages := make([]string, 0, pdf.NumPage())

	for i := 0; i < pdf.NumPage(); i++ {
		pageText, err := pdf.Text(i)
		if err != nil {
			output <- ParsedChunk{Error: err}
			return
		}
		pages = append(pages, pageText)
	}

	output <- ParsedChunk{
		Text:   strings.Join(pages, "\n\n"),
		Offset: 0,
		Error:  nil,
	}
}

func (parser *DefaultParser) parsePlaintext(filename string, data io.Reader, output chan ParsedChunk) {
	offset := 0
	for {
		chunk := make([]byte, parser.maxChunkSize)

		n, err := io.ReadFull(data, chunk)
		isEnd := false
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			err = nil
			isEnd = true
		}

		output <- ParsedChunk{
			Text:   string(chunk[:n]),
			Offset: offset,
			Error:  err,
		}
		offset += n

		if isEnd {
			return
		}
	}
}
