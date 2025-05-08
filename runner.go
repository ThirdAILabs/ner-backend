package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	// Added for timing
	"ner-backend/internal/core"
)

var sentence = `Most Recent I-94
Note to employers, local, state or federal agency granting benefits:
Please visit the CBP I-94 Public Website and click on the tab for Get Most Recent I-94 to
perform a search for the applicant to confirm that the biographic and travel information
displayed on this I-94 printout matches the Get Most Recent I-94 returned results for this
applicant. I-94 FAQs: (https://i94.cbp.dhs.gov/I94/#/faq)
Admission I-94 Record Number:386631062A4
Arrival/Issued Date:2025 February 02
Class of Admission:H1B
Admit Until Date:2026 September 30
Details provided on the I-94 Information form:Last/Surname:GUPTA
First (Given) Name:SHUBH
Birth Date:2000 September 13
Document Number:T3234636
Country of Citizenship:India
Effective April 26, 2013, DHS began automating the admission process. An alien lawfully
admitted or paroled into the U.S. is no longer required to be in possession of a
preprinted Form I-94. A record of admission printed from the CBP website constitutes a
lawful record of admission. See 8 CFR § 1.4(d).
If an employer, local, state or federal agency requests admission information, present
your admission (I-94) number along with any additional required documents requested
by that employer or agency.
Note: For security reasons, we recommend that you close your browser after you have
finished retrieving your I-94 number.
OMB No. 1651-0111 Expiration Date: 03/31/2025
3/25/25, 9:37 AM I-94 Official Website - Get Most Recent I-94 Response
https://i94.cbp.dhs.gov/search/recent-search/results 1/1
`

func run() error {

	model, err := core.LoadModel("cnn", "/home/pratik/infer/ner-models/cnn_model")

	fmt.Println("Model loaded")

	if err != nil {
		fmt.Println("Error loading model", err)
	}

	defer model.Release()

	result, err := model.Predict(sentence)
	if err != nil {
		return err
	}

	fmt.Println(result)

	return nil
}

func main() {
	// We don't want to see the plugin logs.
	log.SetOutput(ioutil.Discard)

	if err := run(); err != nil {
		fmt.Printf("error: %+v\n", err)
		os.Exit(1)
	}

	os.Exit(0)
}
