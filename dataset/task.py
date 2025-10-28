from .classification import *
from .extraction import *
from .generation import *

from typing import Union


def get_type_task(task: str) -> str:
    CLS_CLF = Union[
        Task_clf_binary, Task_clf_mul_class, Task_clf_mul_label, Task_clf_mul_question
    ]

    CLS_EXT = Union[Task_ext]

    CLS_GEN = Union[Task_gen]

    if isinstance(task, CLS_CLF):
        return "classification"
    elif isinstance(task, CLS_EXT):
        return "extraction"
    elif isinstance(task, CLS_GEN):
        return "generation"
    else:
        raise ValueError("Invalid task type")


def get_cls_for_clf():
    # Classification
    return {
        "ADE-Identification": Task_clf_ADE_ADE_identification,
        "BrainMRI-AIS": Task_clf_Brain_MRI_AIS,
        "Brateca-Hospitalization": Task_clf_Brateca_hospitalization,
        "Brateca-Mortality": Task_clf_Brateca_mortality,
        "Cantemist-Coding": Task_nor_Cantemist_CODING,
        "CARES-Area": Task_clf_CARES_area,
        "CARES-ICD10 Chapter": Task_clf_CARES_icd10_chapter,
        "CARES ICD10 Block": Task_nor_CARES_icd10_block,
        "CARES-ICD10 Subblock": Task_nor_CARES_icd10_sub_block,
        "C-EMRS": Task_clf_C_EMRS,
        "ClinicalNotes-UPMC": Task_clf_Clinical_Notes_UPMC,
        "PPTS": Task_clf_clinical_records_from_the_Mexican_Social_Security_Institute,
        "CLIP": Task_clf_CLIP,
        "DialMed": Task_clf_Dial_Med,
        "EHRQA-Primary department": Task_clf_EHRQA_primary_department,
        "EHRQA-Sub department": Task_clf_EHRQA_sub_department,
        "GOUT-CC-Consensus": Task_clf_GOUT_CC_consensus,
        "JP-STS": Task_clf_Japanese_Case_Reports,
        "MEDIQA 2019-RQE": Task_clf_MEDIQA_2019_Task2_RQE,
        "MedNLI": Task_clf_MedNLI,
        "MedSTS": Task_clf_MedSTS,
        "MTS": Task_clf_mtsamples,
        "MEDIQA 2023-sum-A": Task_clf_MTS_Dialog_MEDIQA_2023_sum_task_A,
        "RuMedDaNet": Task_clf_RuMedDaNet,
        "CBLUE-CDN": Task_nor_CHIP_CDN,
        "CHIP-CTC": Task_clf_CHIP_CTC,
        "IMCS-V2-DAC": Task_clf_IMCS_V2_DAC,
        "RuMedNLI": Task_clf_RuMedNLI,
        "CLISTER": Task_clf_CLISTER,
        "IFMIR-Incident type": Task_clf_IFMIR_IncidentType,
        "MIMIC-IV CDM": Task_clf_mimic_iv_CDM,
        "MIMIC-III Outcome.LoS": Task_clf_mimic_iii_outcome_LoS,
        "MIMIC-III Outcome.Mortality": Task_clf_mimic_iii_outcome_Mortality,
        "MIMIC-IV DiReCT.Dis": Task_clf_mimic_iv_DiReCT_Dis,
        "MIMIC-IV DiReCT.PDD": Task_clf_mimic_iv_DiReCT_PDD,
    }


def get_cls_for_ext():
    # Extraction
    return {
        "ADE-Extraction": Task_ext_ADE_ADE_relation,
        "ADE-Drug dosage": Task_ext_ADE_Drug_dosage,
        "BARR2": Task_ext_BARR2_resolution,
        "Cantemis-NER": Task_ext_Cantemist_NER,
        "Cantemis-Norm": Task_ext_Cantemist_Norm,
        "CHIP-CDEE": Task_ext_CHIP_CDEE,
        "CodiEsp-ICD-10-CM": Task_ext_CLEF_ICD_10_CM,
        "CodiEsp-ICD-10-PCS": Task_ext_CLEF_ICD_10_PCS,
        "CLINpt-NER": Task_ext_CLINpt,
        "DiSMed-NER": Task_ext_DiSMed,
        "MIE": Task_ext_MIE,
        "Ex4CDS": Task_ext_Ex4CDS,
        "n2c2 2006-De-identification": Task_ext_n2c2_2006_De_Identification,
        "Medication extraction": Task_ext_i2b2_2009_Medication_Extraction_Challenge,
        "n2c2 2010-Concept": Task_ext_i2b2_2010_Relations_Challenge_concept,
        "n2c2 2010-Assertion": Task_ext_i2b2_2010_Relations_Challenge_assertion,
        "n2c2 2010-Relation": Task_ext_i2b2_2010_Relations_Challenge_relation,
        "n2c2 2014-De-identification": Task_ext_n2c2_2014_De_identification,
        "IMCS-V2-NER": Task_ext_IMCS_V2_NER,
        "meddocan": Task_ext_meddocan,
        "MTS-Temporal": Task_ext_MTSamples_temporal_annotation,
        "n2c2 2018-ADE&medication": Task_ext_n2c2_2018_Track2_Adverse_Drug_Events_and_Medication_Extraction,
        "NorSynthClinical-NER": Task_ext_NorSynthClinical_entity,
        "NorSynthClinical-RE": Task_ext_NorSynthClinical_relation,
        "NUBES": Task_ext_NUBES,
        "CHIP-MDCFNPC": Task_ext_CHIP_MDCFNPC,
        "IMCS-V2-SR": Task_ext_IMCS_V2_SR,
        "n2c2 2014-Diabetes": Task_ext_n2c2_2014_Heart_Disease_Challenge_Diabete,
        "n2c2 2014-CAD": Task_ext_n2c2_2014_Heart_Disease_Challenge_CAD,
        "n2c2 2014-Hyperlipidemia": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hyperlipidemia,
        "n2c2 2014-Hypertension": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hypertension,
        "n2c2 2014-Medication": Task_ext_n2c2_2014_Heart_Disease_Challenge_Medication,
        "CAS-label": Task_ext_CAS_label,
        "RuDReC-NER": Task_ext_RuDReC,
        "NorSynthClinical-PHI": Task_ext_NorSynthClinical_PHI,
        "RuCCoN": Task_ext_RuCCoN_NER,
        "BRONCO150-NER&Status": Task_ext_BRONCO150_NER_status,
        "CARDIO-DE": Task_ext_CARDIO_DE,
        "GraSSCo PHI": Task_ext_GraSSCo_PHI,
        "IFMIR-NER": Task_ext_IFMIR_NER,
        "IFMIR-NER&factuality": Task_ext_IFMIR_NER_factuality,
        "iCorpus": Task_ext_iCorpus,
    }


def get_cls_for_gen():
    # Generation
    return {
        "cMedQA": Task_gen_cMedQA,
        "EHRQA-QA": Task_gen_EHRQA_qa,
        "MEDIQA 2023-chat-A": Task_gen_MTS_Dialog_MEDIQA_2023_chat_task_A,
        "MEDIQA 2023-sum-B": Task_gen_MTS_Dialog_MEDIQA_2023_sum_task_B,
        "MedDG": Task_gen_MedDG,
        "IMCS-V2-MRG": Task_gen_IMCS_V2_MRG,
        "CAS-evidence": Task_gen_CAS_evidence,
        "icliniq-10k": Task_gen_icliniq,
        "HealthCareMagic-100k": Task_gen_HealthCareMagic,
        "MIMIC-IV BHC": Task_gen_mimic_iv_BHC,
    }


def get_cls_for_all_tasks():
    dict_cls_for_clf = get_cls_for_clf()
    dict_cls_for_ext = get_cls_for_ext()
    dict_cls_for_gen = get_cls_for_gen()
    dict_type_task_cls = {
        "classification": dict_cls_for_clf,
        "extraction": dict_cls_for_ext,
        "generation": dict_cls_for_gen,
    }
    dict_all_tasks = {}
    dict_all_tasks.update(dict_cls_for_clf)
    dict_all_tasks.update(dict_cls_for_ext)
    dict_all_tasks.update(dict_cls_for_gen)
    return dict_type_task_cls, dict_all_tasks
