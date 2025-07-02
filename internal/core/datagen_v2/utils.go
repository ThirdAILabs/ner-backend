package datagenv2

import (
	"encoding/csv"
	"os"
)

// Requirements is the set of guidelines for generating annotated sentences.
var Requirements = []string{
	"Some sentences must be grammatically correct and naturally written.",
	"Include fluent sentences structure with realistic syntax and semantics.",
	"Include both formal and informal sentence styles.",
	"Maintain linguistic variety: mix sentence types (declarative, interrogative, exclamatory).",
	"Avoid repetition of sentence patterns, phrases, and vocabulary.",
	"Sentences can be from imaginary or real-world contexts.",
	"Use contextually rich phrases that give meaning to the tagged entities.",
	"Ensure each tag appears in at least X different sentence contexts.",
	"Cover a wide range of surface forms for each tag.",
	"Do not use the same example/entity more than once per tag unless rephrased significantly.",
	"Simulate conversations, tweets, formal documents, and questions.",
	"Vary tone: use professional, casual, emotional, sarcastic, humorous tones across samples.",
	"Use a wide variety of vocabulary, idioms, and expressions.",
	"Incorporate some sentences from different regions or dialects (e.g., British vs. American English).",
	"Provide balanced representation of gender, ethnicity, region, and culture in names and contexts.",
	"Simulate real-world noise like: Spelling mistakes, Missing punctuation, Abbreviations.",
	"Insert some occasional irrelevant tokens or fillers (e.g., \"umm\", \"you know\", \"well\").",
	"Mimic automatic speech transcription quirks in a subset of examples.",
	"Include some sentences where entity boundaries are ambiguous.",
	"Include some nested entities or overlapping meanings if needed for advanced models.",
	"Include some examples with acronyms or abbreviations for named entities (e.g., \"IBM\", \"WHO\").",
	"Include some polysemous words as entities or non-entities (e.g., “Apple” as ORG vs fruit)",
	"Include some confusable formats (e.g., phone number vs ID number vs money).",
	"Include some rare or less common entity examples (e.g., \"Yamoussoukro as a LOCATION\").",
	"Include some adversarial examples where the entity could be mistaken for a different tag.",
	"Add some ungrammatical or ill-formed but understandable constructions (e.g., tweets or SMS).",
}

// WriteToCSV appends or creates a CSV file at path with dataPoints and headers.
func WriteToCSV(path string, dataPoints []map[string]string, headers []string) error {
	mode := os.O_CREATE | os.O_WRONLY
	if _, err := os.Stat(path); err == nil {
		mode |= os.O_APPEND
	} else {
		mode |= os.O_TRUNC
	}
	f, err := os.OpenFile(path, mode, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if mode&os.O_TRUNC != 0 {
		if err := w.Write(headers); err != nil {
			return err
		}
	}
	for _, row := range dataPoints {
		record := make([]string, len(headers))
		for i, h := range headers {
			record[i] = row[h]
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	return nil
}
