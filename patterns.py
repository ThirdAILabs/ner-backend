from presidio_analyzer import RecognizerRegistry, PatternRecognizer
import yaml

registry = RecognizerRegistry()
registry.load_predefined_recognizers()

entities = [
  "LOCATION","US_DRIVER_LICENSE","PHONE_NUMBER","DATE_TIME",
  "EMAIL_ADDRESS","CREDIT_CARD","US_SSN","URL",
  "US_PASSPORT","US_ITIN","US_BANK_NUMBER","IN_PAN",
  "IN_AADHAAR","IN_VEHICLE_REGISTRATION",
]

patterns = []
for rec in registry.get_recognizers(language="en", entities=entities):
    if isinstance(rec, PatternRecognizer):
        patterns.append(rec.to_dict())

with open("recognizers.yaml", "w") as f:
    yaml.dump({"recognizers": patterns}, f)
