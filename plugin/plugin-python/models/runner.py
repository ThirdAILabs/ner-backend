from ensemble.ensemble import CombinedNERModel

model = CombinedNERModel(
    model_path="/app/tmp_models_TODO/ebb55081-52a6-4788-bc51-425cce180acc",
    threshold=0.5,
)

res = model.predict(
    "Transmission Corporation Of Andhra Pradesh Limited Abstract Aptransco – Medical – Introduction of Medical Reimbursement/Credit Card facility for the Aptransco/discoms employees and their dependents/retired employees/dependents/family pensioners - Replacing the existing medical Insurance Scheme – Orders – Issued. T.O.O.(Addl.Secy.-Per) Ms.No.301. Dt: 31-3-2009 Read the following:- T.o.o (Addl.Secy-Per) Ms.No.292, Dt.28-3-2008 T.O.O.(Addl.Secy-Per) Ms.No.158, dt.27-9.2008. Govt. of AP’s Energy (Ser) Dept. Lr.No.1861/Ser/2009, dt.30.3.2009. Order: Consequent to introduction of Standard Group Mediclaim Policy vide references cited in year 2008 w.e.f. 1-4-2008 and with the revision of policy from 1-10-2008 to the inservice employees. The Unions/Association of Aptransco/discoms, were representing for evolving alternate medical reimbursement scheme in place of existing one with the ceiling amounts of the disease on par with the Government of Andhra Pradesh as the present operating scheme of the Insurance Policy is inadequate in meeting the employees requirement. The Aptransco on 17-3-2009 held discussion with the unions/Associations and proposed to implement medical reimbursement scheme/credit card facility with enhancement of ceiling limit in the first instance, till a alternative package deal is evolved: After careful consideration, the Aptransco hereby issue the following orders/Guidelines for implementation of medical reimbursement scheme/availing of credit card facility under the following terms and conditions for Major/Chronic ailments as In-patients."
)
print(res.model_dump_json(indent=2))
