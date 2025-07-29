package main

import (
	"fmt"
	"os"
	"path/filepath"

	"ner-backend/internal/document_parsing"
)

func main() {
	contents, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println("Error reading file:", err)
	}

	switch filepath.Ext(os.Args[1]) {
	case ".docx":
		md, err := document_parsing.DocxToMD(contents)
		if err != nil {
			fmt.Println("Error converting to MD:", err)
		}
		fmt.Println(md)
	case ".pdf":
		md, err := document_parsing.PDFToMD(contents)
		if err != nil {
			fmt.Println("Error converting to PDF:", err)
		}
		fmt.Println(md)
	}
}