// Adapted from https://github.com/koushamad/PDFtoMD/blob/master/PDFtoMD.go

package document_parsing

import (
	"regexp"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/gen2brain/go-fitz"
)

func PDFToMD(contents []byte) (string, error) {
	doc, err := fitz.NewFromMemory(contents)
	if err != nil {
		return "", err
	}
	defer doc.Close()

	numPages := doc.NumPage()
	var mdContent string

	for i := 0; i < numPages; i++ {
		html, err := doc.HTML(i, true)
		if err != nil {
			return "", err
		}

		converter := md.NewConverter("", true, nil)
		text, err := converter.ConvertString(html)
		if err != nil {
			return "", err
		}

		// Remove hardcoded images before adding to content to reduce content size.
		mdContent += removeHardcodedImages(text) + "\n\n"
	}

	return mdContent, nil
}

func removeHardcodedImages(content string) string {
	// Remove hardcoded base64 images in the format ![](data:image/...)
	re := regexp.MustCompile(`!\[\]\(data:image/[^)]+\)`)
	return re.ReplaceAllString(content, "")
}
